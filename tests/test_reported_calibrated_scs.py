"""Calibrated SCS provenance, unchanged author dispatch and historical reports."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from dcs_bench import reported_study as study, timed_radial
from scripts import reproduce_scs_100d as runner


class ReportedCalibratedSCSTests(unittest.TestCase):
    def test_asset_retains_equal_pilot_counts_and_preselected_candidates(self):
        raw = (study.ASSETS / 'joint_calibration.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), study.CALIBRATION_SHA256)
        self.assertNotIn(b'/Users/', raw)
        asset = json.loads(raw)
        self.assertTrue(asset['frozen_before_assessment'])
        budget = asset['budget_per_family_per_dimension']
        self.assertEqual(budget['candidates'] * budget['shared_exact_target_starts'] *
                         budget['transitions_per_start'], 524288)
        self.assertTrue(budget['same_physical_starts_reused_across_families_and_candidates'])
        for dimension, expected in study.CALIBRATED_SCS.items():
            row = asset['dimensions'][str(dimension)]
            self.assertTrue(row['frozen_before_holdout'])
            self.assertTrue(row['holdout_no_retuning'])
            for method in ('DCS-BW', 'SCS'):
                candidates = [r for r in row['candidates'] if r['method'] == method]
                self.assertEqual(len(candidates), 16)
                self.assertEqual(sum(r['transitions'] for r in candidates), 524288)
                selection = row['selections'][method]
                chosen = next(r for r in candidates if r['id'] == selection['candidate_id'])
                self.assertEqual(chosen['score'], max(r['score'] for r in candidates))
                self.assertEqual(chosen['R'], selection['R'])
                self.assertEqual(chosen['h'], selection['h'])
                costs = row['historical_pilot_costs'][method]
                self.assertAlmostEqual(costs['transition_evaluation_seconds'],
                                       sum(r['evaluation_seconds'] for r in candidates))
            self.assertEqual(row['selections']['SCS']['R'], expected['R'])
            self.assertEqual(row['selections']['SCS']['h'], expected['h'])

    def test_each_backend_retains_original_scs_and_distinct_calibrated_map(self):
        for backend in ('jax', 'numba'):
            for dimension in (100, 500):
                with self.subTest(backend=backend, dimension=dimension):
                    config = study.configuration(comparison=backend, target=f'skew-t-{dimension}d')
                    before = copy.deepcopy(config)
                    prepared = study.load_prepared(config)
                    maps = runner.effective_method_maps(config, prepared['maps'])
                    original = prepared['provenance']['original_fitted_maps']['sphere']
                    self.assertEqual(maps['SCS'], original)
                    chosen = copy.deepcopy(original)
                    chosen['scale'] = study.CALIBRATED_SCS[dimension]['R']
                    chosen.update(mu=[0.] * dimension, observer=[0.] * dimension)
                    self.assertEqual(maps['SCS-calibrated'], chosen)
                    for method, scale, policy in [('SCS', .1, 'author_scs_fixed_01'),
                            ('SCS-calibrated', study.CALIBRATED_SCS[dimension]['h'], 'fixed_scs_move_v1')]:
                        execution = runner.method_execution_config(config, method)
                        self.assertEqual(execution['algorithm'], 'SCS')
                        self.assertEqual(timed_radial.resolve_step_tuning(execution), policy)
                        self.assertEqual(timed_radial._initial_proposal_scale(execution), scale)
                        self.assertEqual(execution['warmup'], 1500)
                    self.assertEqual(config, before)
                    self.assertEqual(set(prepared['fitting']), {'ball', 'sphere'})
                    self.assertEqual(prepared['provenance']['new_tuning_seconds'], 0.)
                    self.assertEqual(prepared['provenance']['new_fitting_seconds'], 0.)

    def test_historical_v1_maps_and_provenance_remain_readable(self):
        for backend in ('jax', 'numba'):
            config = study.legacy_configuration(comparison=backend)
            self.assertEqual(config['algorithms'], study.LEGACY_METHODS)
            self.assertEqual(config['protocol'], study.LEGACY_PROTOCOLS[backend])
            prepared = study.load_prepared(config)
            self.assertEqual(set(prepared['maps']), {'ball', 'sphere'})
            self.assertNotIn('joint_calibration', prepared['provenance'])
            self.assertEqual(prepared['provenance']['schema'], 'reported-zero-centre-prepared-inputs-v1')
            summary = dict(config=config, **{k: prepared[k] for k in ('maps', 'fitting')},
                           prepared_input_provenance=prepared['provenance'],
                           method_maps=runner.effective_method_maps(config, prepared['maps']))
            study.validate_reported_summary(summary)

    def test_calibrated_parameters_and_assets_cannot_be_overridden(self):
        original = study.configuration()
        for change in ({'scs_calibrated_move': .1}, {'reported_calibration_asset_sha256': 'changed'},
                       {'calibration_budget_per_family': {}}, {'reported_calibrated_scs_parameters': {}}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / 'must-not-exist'
                with self.assertRaises(ValueError):
                    study.run(dict(original, **change), output)
                self.assertFalse(output.exists())
        prepared = study.load_prepared(original)
        prepared['maps']['sphere_calibrated']['observer'][0] += .1
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, 'exact bundled'):
            runner.run(original, Path(temp) / 'must-not-exist', prepared_inputs=prepared)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in ('manifest.json', 'full_budget_100d.json', 'joint_calibration.json'):
                (directory/name).write_bytes((study.ASSETS/name).read_bytes())
            with (directory/'joint_calibration.json').open('ab') as stream:
                stream.write(b' ')
            with patch.object(study, 'ASSETS', directory), self.assertRaisesRegex(ValueError, 'calibration hash'):
                study.configuration()

    def test_chain_dispatch_keeps_distinct_output_id_and_author_kernel(self):
        config = study.configuration(comparison='numba')
        maps = study.load_prepared(config)['maps']
        barrier = Mock()
        with tempfile.TemporaryDirectory() as temp, patch.object(timed_radial, 'run_timed',
                return_value=(np.ones((2, 2)), dict(mean_all_iterations=[.1, 2.], seconds=.05))) as run:
            job = dict(algorithm='SCS-calibrated', chain=0, seed=901001, wave=0, wave_size=1,
                       wave_cpu_ids=[None], barrier=barrier, config=config,
                       mapping=runner.method_mapping(maps, 'SCS-calibrated'),
                       output=str(Path(temp)/'SCS-calibrated.npz'))
            record = runner.chain_job(job)
            self.assertEqual(run.call_args.args[0], maps['sphere_calibrated'])
            self.assertEqual(run.call_args.args[1]['algorithm'], 'SCS')
            self.assertEqual(run.call_args.args[1]['scs_fixed_move'], .05)
            self.assertEqual(record['algorithm'], 'SCS-calibrated')
            self.assertEqual(record['kernel_algorithm'], 'SCS')
            self.assertTrue(Path(job['output']).exists())
            barrier.abort.assert_not_called()

    def test_original_and_calibrated_controls_follow_exact_author_driver(self):
        # Fixed transitions make this a trajectory check, not a timing benchmark.
        config = study.configuration()
        legacy = study.legacy_configuration()
        maps = study.load_prepared(config)['maps']
        for label, expected_step in [('SCS', .1), ('SCS-calibrated', .05)]:
            actual = runner.method_execution_config(config, label)
            reference = dict(legacy, algorithm='SCS')
            mapping = maps['sphere' if label == 'SCS' else 'sphere_calibrated']
            if label == 'SCS-calibrated':
                reference.update(initial_scale=expected_step, scs_fixed_move=expected_step,
                                 step_tuning_policies={'SCS': 'fixed_scs_move_v1'})
            first = timed_radial._Matched(mapping, actual, 717)
            second = timed_radial._Matched(mapping, reference, 717)
            self.assertEqual(first.scale, expected_step)
            self.assertEqual(second.scale, expected_step)
            self.assertFalse(first.tuning_metadata['step_size_adapted'])
            np.testing.assert_array_equal(first.state, second.state)
            for a, b in zip(first.batch(), second.batch()):
                np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
