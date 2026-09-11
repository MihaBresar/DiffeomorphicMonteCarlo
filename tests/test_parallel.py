"""Exercise real worker processes, CPU allocation and reproducible chain seeds."""
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from dcs_bench.cpu import cpu_info, execution_plan, validate_workers, _ordered_slots
from dcs_bench.engine import chain_seed, environment, load_config, read_chain
from dcs_bench.parallel import launch_chains
from dcs_bench.samplers import run_metropolis
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap


class ParallelTests(unittest.TestCase):
    def test_ryzen_16_core_32_thread_allocation(self):
        cores = [[2*i, 2*i+1] for i in range(16)]
        slots = _ordered_slots(cores)
        self.assertEqual(slots[:16], list(range(0, 32, 2)))
        self.assertEqual(slots[16:], list(range(1, 32, 2)))
        info = {'capacity': 32, 'cpu_ids': slots, 'physical_cores': 16}
        with patch('dcs_bench.cpu.cpu_info', return_value=info):
            plan = execution_plan(32, 50)
        self.assertEqual(len(set(plan['cpu_ids'])), 32)
        self.assertEqual(plan['max_concurrent_chains'], 32)

    def test_worker_failure_stops_and_reaps_siblings(self):
        if cpu_info()['capacity'] < 2:
            self.skipTest('Two CPU slots are required')
        config = load_config('skew-t-100d-fixed', 'quick',
                             {'workers': 2, 'iterations': 100000000, 'warmup': 30, 'runs': 2})
        job = {'target': {'kind': 'skew_t', 'dimension': 3, 'nu': 3, 'alpha': {}},
               'config': config, 'algorithm': 'RWM-Gaussian', 'seconds': None,
               'fingerprint': 'failure-test', 'scenario': 0}
        original_popen = subprocess.Popen
        children = []
        def tracked(*args, **kwargs):
            process = original_popen(*args, **kwargs, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            children.append(process)
            return process
        with tempfile.TemporaryDirectory() as tmp:
            chains = [{'index': 0, 'seed': 1, 'path': str(Path(tmp)/'0.npz')},
                      {'index': 1, 'seed': 'invalid seed', 'path': str(Path(tmp)/'1.npz')}]
            plan = execution_plan(2, 2)
            with patch('dcs_bench.parallel.subprocess.Popen', side_effect=tracked):
                with self.assertRaisesRegex(RuntimeError, 'worker.*failed'):
                    launch_chains(job, chains, Path(tmp)/'jobs', plan)
        self.assertEqual(len(children), 2)
        self.assertTrue(all(child.poll() is not None for child in children))

    def test_worker_limits_and_effective_count(self):
        info = {'capacity': 4, 'cpu_ids': [1, 3, 5, 7]}
        for value in (0, -1, 5, 1.5, True, '2'):
            with self.assertRaises(ValueError):
                validate_workers(value, info)
        with patch('dcs_bench.cpu.cpu_info', return_value=info):
            plan = execution_plan(4, 2)
        self.assertEqual(plan['max_concurrent_chains'], 2)
        self.assertEqual(plan['cpu_ids'], [1, 3])

    def test_two_processes_preserve_seeds_draws_and_checkpoints(self):
        workers = min(2, cpu_info()['capacity'])
        config = load_config('skew-t-100d-fixed', 'quick',
                             {'workers': workers, 'iterations': 12000, 'warmup': 30, 'runs': 5, 'thin': 1})
        spec = {'kind': 'skew_t', 'dimension': 3, 'nu': 3, 'alpha': {}}
        target = Target(spec)
        plan = execution_plan(workers, 5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chains = [{'index': i, 'seed': chain_seed(config, 0, 0, i), 'path': str(root/f'{i}.npz')} for i in range(5)]
            # Even when a fitted map exists, an ordinary RWM baseline must ignore it.
            job = {'target': spec, 'config': config, 'algorithm': 'RWM-Gaussian',
                   'maps': {'ball': BallMap(3, beta=2).parameters()},
                   'seconds': None, 'fingerprint': 'parallel-test', 'scenario': 0}
            launch_chains(job, chains, root/'jobs', plan)
            records = []
            for chain in chains:
                draws, record = read_chain(Path(chain['path']), 'parallel-test')
                expected, _ = run_metropolis(target, 'RWM-Gaussian', None, config, chain['seed'])
                np.testing.assert_array_equal(draws, expected)
                self.assertEqual(record['seed'], chain['seed'])
                self.assertIn(record['execution']['worker_slot'], range(workers))
                self.assertEqual(record['execution']['threads']['OMP_NUM_THREADS'], '1')
                if plan['hardware']['pinning'] in ('linux', 'windows'):
                    self.assertTrue(record['execution']['pinned'])
                    self.assertEqual(record['execution']['cpu_id'], plan['cpu_ids'][record['execution']['worker_slot']])
                records.append(record)
            self.assertEqual(len({r['execution']['pid'] for r in records}), workers)
            # Every submitted chain was claimed exactly once, including a final
            # unbalanced batch and a resume containing sparse chain indices.
            claims = list((root/'jobs').glob('*-queue/*.worker-*.json'))
            self.assertEqual(sorted(json.loads(p.read_text())['index'] for p in claims), list(range(5)))
            # No pending chains must not spawn processes or modify checkpoints.
            before = {p.name: p.read_bytes() for p in root.glob('*.npz')}
            launch_chains(job, [], root/'jobs', plan)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.glob('*.npz')})

    def test_zip_install_does_not_require_git(self):
        with patch('dcs_bench.engine.subprocess.run', side_effect=FileNotFoundError('git')):
            result = environment()
        self.assertIsNone(result['git_commit'])
        self.assertIsNone(result['git_status'])


if __name__ == '__main__':
    unittest.main()
