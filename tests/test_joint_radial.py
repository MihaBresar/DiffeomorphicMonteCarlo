"""Independent checks of joint calibration, untouched kernels and held-out waves."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts import audit_joint_radial as audit
from dcs_bench import samplers
from dcs_bench.samplers import MetropolisKernel
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap


class JointRadialTests(unittest.TestCase):
    def test_independent_phase_seed_blocks_do_not_overlap(self):
        blocks = []
        for phase in audit.SEED_BASES:
            # Exercise the complete permitted index range, exceeding this study.
            seeds = [audit.seed_for(phase, t, a, r, h, c)
                     for t in range(10) for a in range(2) for r in range(8)
                     for h in range(8) for c in (0, 1, 19, 99)]
            self.assertEqual(len(set(seeds)), len(seeds))
            for other in blocks:
                self.assertTrue(set(seeds).isdisjoint(other))
            blocks.append(set(seeds))
        with self.assertRaises(ValueError):
            audit.seed_for('pilot', candidate=8)

    def test_equal_radius_counts_anchors_and_duplicate_vi(self):
        for nu in (2, 3):
            for vi in (1., 1.42, np.sqrt(101), 20.):
                radii = audit.radius_grid(100, nu, vi)
                self.assertEqual(len(set(radii)), 8)
                for radius in (.5, 1., np.sqrt(nu), 2., 4., np.sqrt(101), 20., vi):
                    self.assertIn(radius, radii)
                self.assertEqual(radii, sorted(radii))

    def test_profile_selects_step_at_each_radius_before_comparing_radii(self):
        rows = []
        fitted = {'DCS-BW': 1., 'SCS': 1.42}
        for algorithm, steps in audit.GRIDS.items():
            for radius in audit.radius_grid(100, 3, fitted[algorithm]):
                for scale in steps:
                    score = 100. if radius == 2. and scale == steps[3] else 2. if scale == steps[-1] else 1.
                    rows.append(dict(algorithm=algorithm, R=radius, scale=scale,
                        score=score, valid_selection=True))
        result = audit.select_profiles(rows, fitted, 100)
        self.assertEqual(len(result['profiles']), 16)
        self.assertEqual(len(result['selected']), 8)
        for algorithm, steps in audit.GRIDS.items():
            joint = result['selected'][algorithm+'-joint']
            self.assertEqual((joint['R'], joint['scale']), (2., steps[3]))
            self.assertFalse(joint['step_endpoint'])
            vi = result['selected'][algorithm+'-vi']
            self.assertEqual((vi['R'], vi['scale']), (fitted[algorithm], steps[-1]))
            self.assertTrue(vi['step_endpoint'])
            self.assertEqual(result['selected'][algorithm+'-sqrt-d-plus-one']['R'], np.sqrt(101))

    def test_scoring_keeps_unequal_lengths_and_divides_summed_ess_by_measured_time(self):
        arrays = [np.column_stack([np.arange(n) % 2, np.arange(n)+1.]) for n in (20, 40)]
        seen = []
        def ess(values, method):
            seen.append(values.shape)
            return values.size/2
        with patch('arviz.ess', side_effect=ess), patch('arviz.rhat', return_value=1.) as rhat:
            result = audit.score_pilot(arrays, [.5, .75])
        self.assertEqual(seen, [(1, 20), (1, 40), (1, 20), (1, 40)])
        self.assertEqual(result['score'], 30/1.25)
        self.assertEqual(result['iterations_per_chain'], [20, 40])
        self.assertEqual(rhat.call_args.args[0].shape, (2, 20))
        self.assertTrue(result['valid_selection'])
        arrays[0][:, 0] = 0
        invalid = audit.score_pilot(arrays, [.5, .75])
        self.assertFalse(invalid['valid_selection'])
        self.assertEqual(invalid['score'], 0.)
        self.assertIsNone(invalid['bulk_ess']['tail_probability'])

    def test_probe_preserves_every_canonical_decision_and_partitions_rejections(self):
        spec = {'kind': 'skew_t', 'dimension': 5, 'nu': 3., 'alpha': {}}
        target = Target(spec)
        original = samplers.cap_stepout
        for algorithm, mapping, scale in [('DCS-BW', BallMap(5), .3), ('SCS', SphereMap(5), 4.)]:
            baseline = MetropolisKernel(target, algorithm, mapping, 7239)
            observed = audit.BoundaryKernel(target, algorithm, mapping, 7239)
            totals = {key: 0 for key in audit.PROBE_COUNTS}
            for _ in range(600):
                _, accepted = baseline.step(scale)
                counts = audit.boundary_probe(observed, scale, 1)
                self.assertEqual(counts['accepted'], int(accepted))
                np.testing.assert_array_equal(observed.state, baseline.state)
                self.assertEqual(observed.lp, baseline.lp)
                self.assertEqual(observed.rng.bit_generator.state, baseline.rng.bit_generator.state)
                for key in totals:
                    totals[key] += counts[key]
            self.assertIs(samplers.cap_stepout, original)
            self.assertEqual(sum(totals[k] for k in ('accepted', 'final_infeasible', 'nonfinite_density', 'density_rejections')), 600)
            self.assertGreater(totals['initial_boundary_crossings'], 0)
            if algorithm == 'DCS-BW':
                self.assertEqual(totals['initial_boundary_crossings'], totals['final_infeasible'])
                self.assertEqual(totals['stepout_calls'], 0)
            else:
                self.assertEqual(totals['initial_boundary_crossings'], totals['stepout_calls'])
                self.assertGreater(totals['accepted_after_stepout'], 0)

    def test_production_waves_are_homogeneous_and_cover_equal_chain_allocations(self):
        for runs, workers in [(20, 8), (2, 2), (4, 1)]:
            schedule = audit.production_schedule({'runs': runs}, workers)
            for target_id in audit.TARGETS:
                for case in audit.case_definitions(target_id):
                    entries = [w for w in schedule if w['target_id'] == target_id and w['case'] == case['case']]
                    self.assertEqual([c for w in entries for c in w['chains']], list(range(runs)))
            for wave in range((runs+workers-1)//workers):
                entries = [w for w in schedule if w['wave'] == wave]
                self.assertEqual(len(entries), 16)
                self.assertTrue(all(w['chains'] == entries[0]['chains'] for w in entries))

    def test_pilot_waves_alternate_methods_without_mixing_partial_waves(self):
        jobs = [dict(algorithm=a, id=i) for a in audit.GRIDS for i in range(11)]
        waves = audit.pilot_waves(jobs, 4, 0)
        self.assertEqual([w[0]['algorithm'] for w in waves], ['DCS-BW', 'SCS', 'SCS', 'DCS-BW', 'DCS-BW', 'SCS'])
        self.assertEqual(list(map(len, waves)), [4, 4, 4, 4, 3, 3])
        for wave in waves:
            self.assertEqual(len({j['algorithm'] for j in wave}), 1)
        self.assertEqual(sorted((j['algorithm'], j['id']) for w in waves for j in w),
                         sorted((j['algorithm'], j['id']) for j in jobs))
        self.assertEqual(waves, audit.pilot_waves(jobs, 4, 0))

    def test_worker_matches_frozen_kernel_and_enforces_provenance(self):
        spec = {'kind': 'skew_t', 'dimension': 4, 'nu': 2., 'alpha': {'0': 2.}}
        with tempfile.TemporaryDirectory() as directory:
            for algorithm in audit.GRIDS:
                mapping = BallMap(4, scale=1.3) if algorithm == 'DCS-BW' else SphereMap(4, scale=1.3)
                job = dict(phase='production', target=spec, target_id='test', algorithm=algorithm,
                    case=algorithm+'-joint', chain=0, seed=453, R=1.3, scale=.1,
                    map=mapping.parameters(), warmup=10, seconds=14., thin=3, pretuned=True,
                    fingerprint='test', output=str(Path(directory)/(algorithm+'.npz')))
                with self.assertRaisesRegex(ValueError, 'pretuned'):
                    audit.chain_job({**job, 'pretuned': False})
                kernel = MetropolisKernel(Target(spec), algorithm, mapping, job['seed'])
                for _ in range(job['warmup']):
                    kernel.step(job['scale'])
                expected = []
                for _ in range(14):
                    kernel.step(job['scale'])
                    radius = np.linalg.norm(kernel.draw())
                    expected.append([float(radius >= 50), radius])
                collector = audit.collect_timed
                def deterministic(kernel, scale, seconds, thin):
                    ticks = iter([0., *map(float, range(15)), 14.])
                    return collector(kernel, scale, seconds, thin, clock=lambda: next(ticks), cpu_clock=lambda: 0.)
                with patch.object(audit, 'collect_timed', side_effect=deterministic):
                    audit.chain_job(job)
                values, record = audit.load_chain(job)
                np.testing.assert_array_equal(values, np.asarray(expected)[2::3])
                np.testing.assert_allclose(list(record['estimates'].values()), np.mean(expected, axis=0), rtol=0, atol=0)
                for change in ({'R': 2.}, {'seed': 9}, {'scale': .2}, {'map': {**job['map'], 'scale': 2.}}):
                    with self.assertRaises(ValueError):
                        audit.load_chain({**job, **change})


if __name__ == '__main__':
    unittest.main()
