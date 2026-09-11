import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts import audit_symmetric_t as audit
from dcs_bench.samplers import MetropolisKernel, run_metropolis
from dcs_bench.transforms import BallMap, SphereMap
from dcs_bench.targets import Target


class SymmetricAuditTests(unittest.TestCase):
    def job(self, output, algorithm, policy):
        mapping = BallMap(100) if algorithm == 'DCS-BW' else SphereMap(100)
        return {'case': 'untuned-'+algorithm+'-'+policy, 'algorithm': algorithm,
                'map_kind': 'untuned', 'policy': policy, 'map': mapping.parameters(),
                'chain': 0, 'seed': 5500001, 'warmup': 80, 'iterations': 500,
                'thin': 5, 'q95': 29., 'scale': .1, 'fingerprint': 'test',
                'output': str(output)}

    def test_fixed_pilot_candidates_match_direct_kernels_for_both_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            for algorithm in ('DCS-BW', 'SCS'):
                job = self.job(Path(directory)/(algorithm+'.npz'), algorithm, 'fixed_grid')
                target = Target(audit.TARGET)
                self.assertFalse(np.any(target.alpha))
                mapping = (BallMap if algorithm == 'DCS-BW' else SphereMap)(**job['map'])
                kernel = MetropolisKernel(target, algorithm, mapping, job['seed'])
                for _ in range(job['warmup']):
                    kernel.step(.1)
                expected, all_expected = [], []
                for i in range(1, job['iterations']+1):
                    kernel.step(.1)
                    values = audit.functional_values(kernel.draw(), job['q95'])
                    all_expected.append(values)
                    if i % job['thin'] == 0:
                        expected.append(values)
                result = audit.chain_job(job)
                with np.load(job['output'], allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['functions'], expected)
                self.assertEqual(result['scale'], .1)
                self.assertFalse(result['adaptation_during_sampling'])
                np.testing.assert_allclose(
                    [result['all_iteration_estimates'][n] for n in audit.FUNCTIONS],
                    np.mean(all_expected, axis=0), rtol=1e-14, atol=1e-14)
                self.assertEqual(result['all_iteration_count'], job['iterations'])

    def test_warmup_is_applied_to_the_supplied_map_and_matches_canonical_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self.job(Path(directory)/'chain.npz', 'DCS-BW', 'warmup_frozen')
            job['map'] = BallMap(100, scale=1.6).parameters()
            draws, metadata = run_metropolis(Target(audit.TARGET), 'DCS-BW',
                BallMap(**job['map']), job, job['seed'])
            result = audit.chain_job(job)
            with np.load(job['output'], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['functions'],
                    [audit.functional_values(x, job['q95']) for x in draws])
            self.assertEqual(result['scale'], metadata['frozen_scale'])
            self.assertEqual(result['acceptance'], metadata['acceptance_rate'])

    def test_summary_rejects_wrong_seed_or_map(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self.job(Path(directory)/'chain.npz', 'DCS-BW', 'fixed_grid')
            audit.chain_job(job)
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                audit.summarize_jobs([{**job, 'seed': job['seed']+1}], audit.reference())
            changed = {**job, 'map': {**job['map'], 'scale': 2.}}
            with self.assertRaisesRegex(ValueError, 'map mismatch'):
                audit.summarize_jobs([changed], audit.reference())


if __name__ == '__main__':
    unittest.main()
