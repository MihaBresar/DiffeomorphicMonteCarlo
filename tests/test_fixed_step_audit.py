"""Keep the diagnostic bridge faithful to its declared production protocols."""
import hashlib
import ast
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scripts import audit_fixed_steps as audit
from scripts.legacy_fixed_step_loader import LEGACY
from dcs_bench.samplers import MetropolisKernel, run_numpy_metropolis
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap


class FixedStepAuditTests(unittest.TestCase):
    def test_main_prints_original_vi_banner_with_cp1252_redirected_streams(self):
        class StopBeforeOptimization(Exception):
            pass

        def stop_at_jit(function):
            raise StopBeforeOptimization()

        # Execute the unmodified historical VI function, stopping at its first
        # JIT decorator after its real Unicode banner; no optimizer is needed.
        tree = ast.parse(LEGACY.read_text(encoding='utf-8'))
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'optimize_prwm_hyperparameters')
        namespace = {'jax': SimpleNamespace(jit=stop_at_jit)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(LEGACY), 'exec'), namespace)

        def original_banner(*unused):
            try:
                namespace['optimize_prwm_hyperparameters']()
            except StopBeforeOptimization:
                import sys
                print('Diagnostic parameter delta: δ', file=sys.stderr)
                raise

        stdout_bytes, stderr_bytes = io.BytesIO(), io.BytesIO()
        stdout = io.TextIOWrapper(stdout_bytes, encoding='cp1252', errors='strict', write_through=True)
        stderr = io.TextIOWrapper(stderr_bytes, encoding='cp1252', errors='strict', write_through=True)
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(stdout), redirect_stderr(stderr), \
                    patch.object(audit, 'environment', return_value={}), \
                    patch.object(audit, 'prepare', side_effect=original_banner):
                with self.assertRaises(StopBeforeOptimization):
                    audit.main(['--output-dir', directory, '--workers', '1'])
        stdout.flush()
        stderr.flush()
        self.assertIn('Location μ, Scale R, Möbius Shift δ', stdout_bytes.getvalue().decode('utf-8'))
        self.assertIn('Diagnostic parameter delta: δ', stderr_bytes.getvalue().decode('utf-8'))
        stdout.close()
        stderr.close()

    def test_original_input_bytes_preserved(self):
        self.assertEqual(hashlib.sha256(LEGACY.read_bytes()).hexdigest(),
                         '72831b9d8b669f2a8705eb0e3fd3c03cb5f2d87a1660b1b2036d6d51601c8a70')

    def make_job(self, output, policy):
        return {'case': f'current-DCS-BW-{policy}', 'source': 'current',
            'algorithm': 'DCS-BW', 'policy': policy, 'chain': 0, 'seed': 901001,
            'target': {'kind': 'skew_t', 'dimension': 100, 'nu': 3, 'alpha': {'0': 100., '1': -100.}},
            'map': BallMap(100, beta=2., scale=1.8).parameters(),
            'warmup': 80, 'iterations': 1200, 'thin': 10,
            'fingerprint': 'scientific-source-test', 'setup_digest': 'frozen-map-test',
            'output': str(output)}

    @staticmethod
    def expected_functionals(draws):
        norm = np.linalg.norm(draws, axis=1)
        return np.column_stack([norm >= 50., norm, np.log1p(norm), draws[:, :4]])

    def test_raw_frozen_matches_canonical_production_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'chain.npz'
            job = self.make_job(path, 'raw_frozen')
            draws, expected = run_numpy_metropolis(Target(job['target']), 'DCS-BW', BallMap(**job['map']), job, job['seed'])
            audit.chain_job(job)
            with np.load(path, allow_pickle=False) as result:
                np.testing.assert_allclose(result['functionals'], self.expected_functionals(draws), rtol=1e-14, atol=1e-14)
                record = json.loads(str(result['metadata']))
            self.assertEqual(record['final_scale'], expected['frozen_scale'])
            self.assertEqual(record['acceptance'], expected['acceptance_rate'])
            self.assertFalse(record['adaptation_during_sampling'])
            with self.assertRaises(FileExistsError):
                audit.chain_job(job)

    def test_independent_pilot_scale_stays_fixed_through_warmup_and_production(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'chain.npz'
            job = self.make_job(path, 'fixed_pilot')
            job['fixed_scale'] = .15
            kernel = MetropolisKernel(Target(job['target']), 'DCS-BW', BallMap(**job['map']), job['seed'])
            for _ in range(job['warmup']):
                kernel.step(.15)
            draws = []
            for iteration in range(1, job['iterations']+1):
                kernel.step(.15)
                if iteration % job['thin'] == 0:
                    draws.append(kernel.draw())
            audit.chain_job(job)
            with np.load(path, allow_pickle=False) as result:
                np.testing.assert_allclose(result['functionals'], self.expected_functionals(np.asarray(draws)), rtol=1e-14, atol=1e-14)
                record = json.loads(str(result['metadata']))
            self.assertEqual(record['warmup_scale'], .15)
            self.assertEqual(record['final_scale'], .15)
            self.assertFalse(record['adaptation_during_sampling'])

    def test_summary_rejects_another_runs_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = self.make_job(root/'chains/original-DCS-BW-raw_frozen/000.npz', 'raw_frozen')
            job.update(case='original-DCS-BW-raw_frozen', source='original')
            audit.chain_job(job)
            audit.write_json(root/'maps.json', {'config': {
                'targets': [job['target']], 'preset': 'quick', 'runs': 1,
                'iterations': job['iterations'], 'thin': job['thin']},
                'fingerprint': 'different-scientific-source', 'setup_digest': job['setup_digest']})
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                audit.summarize(root)

    def test_heldout_seeds_reach_jobs_and_summary_validation(self):
        config = audit.audit_config('quick', 1, production_seed_base=9901001)
        config.update(iterations=1280, thin=10)
        ball, sphere = BallMap(100).parameters(), SphereMap(100).parameters()
        fits = {'config': config, 'fingerprint': 'heldout-source', 'setup_digest': 'heldout-maps',
            'maps': {'original': {'ball': ball, 'sphere': sphere},
                     'current': {'ball': ball, 'sphere': sphere}, 'vanilla': {'ball': ball}},
            'pilot_selection': {key: {'scale': .1, 'valid_selection': True, 'endpoint_winner': False}
                                for key in ('mobius-DCS-BW', 'current-SCS', 'vanilla-DCS-BW')}}
        rng = np.random.default_rng(21761)
        arrays = rng.normal(size=(2, 128, 7))
        arrays[:, :, 0] = rng.random((2, 128)) < .2
        arrays[:, :, 1] = np.exp(arrays[:, :, 1])
        arrays[:, :, 2] = np.log1p(arrays[:, :, 1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = audit.make_jobs(root, config, fits)
            self.assertEqual({job['seed'] for job in jobs}, {9901001, 9901002})
            self.assertTrue(all(job['production_seed_base'] == 9901001 for job in jobs))
            audit.write_json(root/'maps.json', fits)
            for job in jobs:
                path = Path(job['output'])
                path.parent.mkdir(parents=True, exist_ok=True)
                values = arrays[job['chain']]
                record = {**job, 'mean': values.mean(axis=0).tolist(), 'acceptance': .25,
                          'warmup_scale': .1, 'final_scale': .1}
                np.savez_compressed(path, functionals=values, metadata=np.array(json.dumps(record)))
            with redirect_stdout(io.StringIO()):
                summary = audit.summarize(root)
            first_case = summary['cases'][jobs[0]['case']]
            self.assertEqual(first_case['diagnostic_draw_counts']['total_retained'], 256)
            self.assertEqual(first_case['iterations_per_chain'], 1280)
            self.assertEqual(first_case['diagnostic_draw_counts']['retained_per_chain'], 128)
            expected = audit.functional_diagnostics(arrays)
            self.assertEqual(first_case['bulk_ess'], expected['bulk_ess'])
            # A default-seed chain cannot silently enter the held-out cohort.
            path = Path(jobs[0]['output'])
            with np.load(path, allow_pickle=False) as data:
                record = json.loads(str(data['metadata']))
                values = data['functionals'].copy()
            record['seed'] = 901001
            np.savez_compressed(path, functionals=values, metadata=np.array(json.dumps(record)))
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                audit.summarize(root)

    def test_rank_diagnostics_match_heldout_functions_and_reject_invalid_traces(self):
        import arviz as az
        rng = np.random.default_rng(29918)
        values = rng.normal(size=(3, 500, 7))
        values[:, :, 0] = rng.random((3, 500)) < .1
        result = audit.functional_diagnostics(values)
        contrast = np.arcsinh((values[:, :, 3]-values[:, :, 4])/np.sqrt(2))
        expected_ess = float(az.ess(contrast, method='bulk'))
        self.assertEqual(result['bulk_ess']['asinh_contrast'], expected_ess)
        self.assertEqual(result['rank_rhat']['asinh_contrast'], float(az.rhat(contrast, method='rank')))
        self.assertEqual(result['bulk_ess_per_retained_draw']['asinh_contrast'], expected_ess/1500)
        values[0, :, 0] = 0
        values[0, 0, 5] = np.nan
        invalid = audit.functional_diagnostics(values)
        for name in ('tail_50', 'asinh_x3', 'coordinate_3'):
            self.assertIsNone(invalid['bulk_ess'][name])
            self.assertIsNone(invalid['rank_rhat'][name])
            self.assertIsNone(invalid['bulk_ess_per_retained_draw'][name])
            self.assertIn(name, invalid['unavailable_diagnostics'])
        self.assertIsNotNone(invalid['bulk_ess']['asinh_x4'])


if __name__ == '__main__':
    unittest.main()
