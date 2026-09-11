"""Resume only an exact source migration, without rewriting saved chains."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from dcs_bench import engine


class ResumeCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root/'run'
        self.directory = self.output/'scenario-0'
        self.directory.mkdir(parents=True)
        (self.root/'validation').mkdir()
        self.config = {'workers': 1, 'runs': 2, 'seed': 123, 'mode': 'iterations',
                       'algorithms': ['RWM-Gaussian'], 'targets': [{'kind': 'stub'}]}
        self.before = {'dcs_bench/worker.py': 'a'*64, 'dcs_bench/engine.py': 'b'*64,
                       'dcs_bench/samplers.py': 'c'*64, 'dcs_bench/plotting.py': 'd'*64}
        self.after = {**self.before, 'dcs_bench/worker.py': 'e'*64, 'dcs_bench/engine.py': 'f'*64}
        self.saved_environment = {'python': 'fixture-python', 'packages': {'numpy': 'fixture'}}
        self.original_digest = engine.fingerprint(self.config, self.before)
        self.original_manifest = {'fingerprint': self.original_digest, 'source_hashes': self.before,
                                  'environment': self.saved_environment, 'status': 'failed'}
        engine.write_json(self.output/'configuration.json', self.config)
        engine.write_json(self.output/'manifest.json', self.original_manifest)
        engine.write_json(self.directory/'reference.json', {})
        engine.write_json(self.directory/'setup.json', {})
        self.saved_chain = self.directory/'chains/RWM-Gaussian-000.npz'
        engine.save_chain(self.saved_chain, np.arange(8).reshape(4, 2),
                          {'fingerprint': self.original_digest, 'chain': 0})
        self.saved_bytes = self.saved_chain.read_bytes()
        self.transition = {'id': 'queue-fixture', 'reason': 'Scheduling fix with unchanged sampler and seeds.',
                           'source_changes': {name: {'before': self.before[name], 'after': self.after[name]}
                                              for name in ('dcs_bench/worker.py', 'dcs_bench/engine.py')}}
        engine.write_json(self.root/'validation/resume_compatibility.json',
                          {'schema_version': 1, 'transitions': [self.transition]})
        self.launched = []

    def fake_launch(self, job, pending, directory, plan):
        self.launched.append((job, pending))
        for chain in pending:
            engine.save_chain(Path(chain['path']), np.arange(8).reshape(4, 2),
                              {'fingerprint': job['fingerprint'], 'chain': chain['index'],
                               'producing_source_fingerprint': job['producing_source_fingerprint'],
                               'producing_source_hashes': job['producing_source_hashes']})

    def resume(self, *, config=None, hashes=None, environment=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(engine, 'ROOT', self.root))
            stack.enter_context(patch.object(engine, 'source_hashes', return_value=hashes or self.after))
            stack.enter_context(patch.object(engine, 'environment', return_value=environment or self.saved_environment))
            stack.enter_context(patch.object(engine, 'execution_plan', return_value={'max_concurrent_chains': 1}))
            stack.enter_context(patch.object(engine, 'Target', return_value=object()))
            stack.enter_context(patch.object(engine, 'launch_chains', side_effect=self.fake_launch))
            stack.enter_context(patch.dict('sys.modules', {'dcs_bench.plotting': SimpleNamespace(rebuild=Mock())}))
            return engine.run(config or self.config, self.output, resume=True)

    def assert_rejected_without_writes(self, **kwargs):
        before = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        with self.assertRaises(ValueError):
            self.resume(**kwargs)
        after = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(self.launched, [])

    def test_registered_migration_reuses_chain_bytes_and_records_actual_sources(self):
        self.resume()
        self.assertEqual(self.saved_bytes, self.saved_chain.read_bytes())
        self.assertEqual([p['index'] for p in self.launched[0][1]], [1])
        new_digest = engine.fingerprint(self.config, self.after)
        _, record = engine.read_chain(self.directory/'chains/RWM-Gaussian-001.npz', self.original_digest)
        self.assertEqual(record['producing_source_fingerprint'], new_digest)
        self.assertEqual(record['producing_source_hashes'], self.after)
        self.resume()
        self.assertEqual(self.launched[1][1], [])
        self.assertEqual(self.saved_bytes, self.saved_chain.read_bytes())
        manifest = json.loads((self.output/'manifest.json').read_text())
        self.assertEqual(manifest['fingerprint'], self.original_digest)
        self.assertEqual(manifest['source_hashes'], self.before)
        self.assertEqual(len(manifest['resume_source_updates']), 1)
        self.assertEqual(manifest['resume_source_updates'][0]['source_changes'], self.transition['source_changes'])
        self.assertEqual(len(manifest['invocations']), 2)
        for invocation in manifest['invocations']:
            self.assertEqual(invocation['producing_source_fingerprint'], new_digest)
            self.assertEqual(invocation['producing_source_hashes'], self.after)

    def test_unknown_source_change_is_rejected(self):
        self.assert_rejected_without_writes(hashes={**self.after, 'dcs_bench/samplers.py': '0'*64})

    def test_wrong_before_hash_is_rejected(self):
        changed = copy.deepcopy(self.original_manifest)
        changed['source_hashes']['dcs_bench/worker.py'] = '0'*64
        changed['fingerprint'] = engine.fingerprint(self.config, changed['source_hashes'])
        engine.write_json(self.output/'manifest.json', changed)
        self.assert_rejected_without_writes()

    def test_modified_configuration_is_rejected(self):
        self.assert_rejected_without_writes(config={**self.config, 'seed': 124})

    def test_modified_environment_is_rejected_before_migration_is_recorded(self):
        self.assert_rejected_without_writes(environment={**self.saved_environment, 'python': 'different-python'})

    def test_forged_saved_fingerprint_is_rejected(self):
        engine.write_json(self.output/'manifest.json', {**self.original_manifest, 'fingerprint': '0'*64})
        self.assert_rejected_without_writes()

    def test_plotting_only_change_needs_no_migration(self):
        (self.root/'validation/resume_compatibility.json').unlink()
        self.resume(hashes={**self.before, 'dcs_bench/plotting.py': '0'*64})
        manifest = json.loads((self.output/'manifest.json').read_text())
        self.assertNotIn('resume_source_updates', manifest)
        self.assertEqual(self.saved_bytes, self.saved_chain.read_bytes())

    def test_unregistered_change_requires_new_run(self):
        (self.root/'validation/resume_compatibility.json').unlink()
        self.assert_rejected_without_writes()


if __name__ == '__main__':
    unittest.main()
