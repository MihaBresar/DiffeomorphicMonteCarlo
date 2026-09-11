"""Timing, adaptive control, and provenance checks for the equal-time audit."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import audit_skew_t_timed as audit
from dcs_bench.samplers import MetropolisKernel
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap


class LegacyAverage:
    """Independent transcription of worker_pbw's historical recurrence."""
    def __init__(self, scale):
        self.mu, self.hbar, self.logbar, self.i = np.log(10*scale), 0., 0., 0

    def update(self, probability):
        self.i += 1
        weight = 1/(self.i+10.)
        self.hbar = (1-weight)*self.hbar+weight*(.234-probability)
        instantaneous = self.mu-np.sqrt(self.i)/.05*self.hbar
        average_weight = self.i**(-.75)
        self.logbar = average_weight*instantaneous+(1-average_weight)*self.logbar
        return float(np.exp(np.clip(self.logbar, -30, 15)))


class TimedAuditTests(unittest.TestCase):
    def job(self, output, algorithm='DCS-BW', policy='frozen'):
        target = {'kind': 'skew_t', 'dimension': 4, 'nu': 2., 'alpha': {'0': 2.}}
        mapping = (BallMap(4, scale=1.3, mu=np.array([.1, 0., 0., 0.]))
                   if algorithm == 'DCS-BW' else SphereMap(4, scale=1.3))
        case = next(c for c in audit.case_definitions()
                    if c['algorithm'] == algorithm and c['policy'] == policy)
        return {**case, 'phase': 'production', 'target': target, 'chain': 0,
                'seed': 78934, 'map': mapping.parameters(), 'scale': .1,
                'pretuned': True, 'warmup': 12, 'seconds': 14., 'thin': 3,
                'fingerprint': 'test-run', 'output': str(output)}

    def test_deadline_overshoot_and_every_rejected_repeat_contribute(self):
        class ScriptedKernel:
            def __init__(self):
                self.wall, self.cpu, self.radius = 0., 0., 1.
                self.evaluations, self.i, self.scales = 3, 0, []

            def draw(self):
                return np.array([self.radius, 0.])

            def step(self, scale):
                self.scales.append(scale)
                accepted = self.i % 3 == 0
                if accepted:
                    self.radius = [51., 3., 101.][self.i//3]
                self.wall += .25
                self.cpu += .1
                self.evaluations += 1
                self.i += 1
                return .8 if accepted else .2, accepted

        kernel = ScriptedKernel()
        values, record = audit.collect_timed(kernel, .12, 2.1, 2,
            clock=lambda: kernel.wall, cpu_clock=lambda: kernel.cpu)
        all_radii = np.array([51., 51., 51., 3., 3., 3., 101., 101., 101.])
        expected = np.column_stack([all_radii >= 50, all_radii])
        np.testing.assert_array_equal(values, expected[1::2])
        np.testing.assert_allclose([record['estimates'][n] for n in audit.METRICS],
                                   expected.mean(axis=0), rtol=0, atol=0)
        self.assertNotEqual(record['estimates']['mean_norm'], values[:, 1].mean())
        self.assertEqual(record['iterations'], 9)
        self.assertEqual(record['retained'], 4)
        self.assertEqual(record['density_evaluations'], 9)
        self.assertEqual(record['sampling_seconds'], 2.25)
        self.assertAlmostEqual(record['overshoot_seconds'], .15)
        self.assertAlmostEqual(record['cpu_seconds'], .9)
        self.assertEqual(record['production_adaptation_updates'], 0)
        self.assertEqual(kernel.scales, [.12]*9)

    def test_frozen_and_adaptive_chains_match_direct_kernel_and_legacy_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            for algorithm in ('DCS-BW', 'SCS'):
                for policy in ('frozen', 'adaptive'):
                    with self.subTest(algorithm=algorithm, policy=policy):
                        job = self.job(Path(directory)/(algorithm+'-'+policy+'.npz'), algorithm, policy)
                        mapping = (BallMap if algorithm == 'DCS-BW' else SphereMap)(**job['map'])
                        kernel = MetropolisKernel(Target(job['target']), algorithm, mapping, job['seed'])
                        adapter = LegacyAverage(job['scale']) if policy == 'adaptive' else None
                        scale = job['scale']
                        for _ in range(job['warmup']):
                            probability, _ = kernel.step(scale)
                            if adapter is not None:
                                scale = adapter.update(probability)
                        production_initial = scale
                        expected, scales = [], []
                        for _ in range(int(job['seconds'])):
                            scales.append(scale)
                            probability, _ = kernel.step(scale)
                            if adapter is not None:
                                scale = adapter.update(probability)
                            expected.append(audit.radial_values(kernel.draw()))
                        # Save the original collector before patching its name.
                        collector = audit.collect_timed
                        def deterministic(kernel, scale, seconds, thin, adapter=None):
                            ticks = iter([0., *map(float, range(int(seconds)+1)), float(seconds)])
                            return collector(kernel, scale, seconds, thin, clock=lambda: next(ticks),
                                             cpu_clock=lambda: 0., adapter=adapter)
                        with patch.object(audit, 'collect_timed', side_effect=deterministic):
                            result = audit.chain_job(job)
                        values, saved = audit.load_chain(job)
                        np.testing.assert_array_equal(values, np.asarray(expected)[job['thin']-1::job['thin']])
                        np.testing.assert_allclose([result['estimates'][n] for n in audit.METRICS],
                                                   np.mean(expected, axis=0), rtol=0, atol=0)
                        self.assertEqual(result['pilot_scale'], job['scale'])
                        self.assertEqual(result['scale'], production_initial)
                        self.assertEqual(result['final_scale'], scale)
                        self.assertEqual(result['min_proposal_scale'], min(scales))
                        self.assertEqual(result['max_proposal_scale'], max(scales))
                        self.assertEqual(result['adaptation_during_sampling'], policy == 'adaptive')
                        self.assertEqual(result['production_adaptation_updates'], 14 if policy == 'adaptive' else 0)
                        self.assertEqual(saved['warmup_adaptation_updates'], 12 if policy == 'adaptive' else 0)

    def test_waves_cover_each_case_with_identical_load_and_rotated_order(self):
        names = [case['case'] for case in audit.case_definitions()]
        self.assertEqual(len(names), 8)
        for runs, workers in [(20, 8), (9, 1), (2, 2)]:
            schedule = audit.production_schedule({'runs': runs}, workers)
            first_cases = []
            for wave in range((runs+workers-1)//workers):
                entries = [entry for entry in schedule if entry['wave'] == wave]
                expected_chains = list(range(wave*workers, min((wave+1)*workers, runs)))
                actual_order = [e['case'] for e in entries]
                self.assertIn(actual_order, [names[k:]+names[:k] for k in range(len(names))])
                first_cases.append(actual_order[0])
                self.assertTrue(all(e['chains'] == expected_chains for e in entries))
            self.assertEqual(len(set(first_cases)), min(len(first_cases), len(names)))
            for case in names:
                self.assertEqual([chain for e in schedule if e['case'] == case for chain in e['chains']],
                                 list(range(runs)))
        with self.assertRaisesRegex(ValueError, 'one job per worker'):
            audit.run_jobs([{}, {}], 1, {'cpu_ids': [0]}, synchronize=True)

    def test_production_requires_pretuning_and_provenance_matches_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self.job(Path(directory)/'chain.npz')
            with self.assertRaisesRegex(ValueError, 'pretuned'):
                audit.chain_job({**job, 'pretuned': False})
            with self.assertRaisesRegex(ValueError, 'policy'):
                audit.chain_job({**job, 'policy': 'unknown'})
            collector = audit.collect_timed
            def deterministic(kernel, scale, seconds, thin, adapter=None):
                ticks = iter([0., *map(float, range(int(seconds)+1)), float(seconds)])
                return collector(kernel, scale, seconds, thin, clock=lambda: next(ticks),
                                 cpu_clock=lambda: 0., adapter=adapter)
            with patch.object(audit, 'collect_timed', side_effect=deterministic):
                audit.chain_job(job)
            for change in [{'seed': job['seed']+1}, {'policy': 'adaptive'}, {'thin': 2},
                           {'scale': .2}, {'map': {**job['map'], 'scale': 2.}}]:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    audit.load_chain({**job, **change})
            with np.load(job['output'], allow_pickle=False) as saved:
                values, record = saved['functions'].copy(), json.loads(str(saved['metadata']))
            record['adaptation_during_sampling'] = True
            np.savez_compressed(job['output'], functions=values, metadata=np.array(json.dumps(record)))
            with self.assertRaisesRegex(ValueError, 'policy mismatch'):
                audit.load_chain(job)


if __name__ == '__main__':
    unittest.main()
