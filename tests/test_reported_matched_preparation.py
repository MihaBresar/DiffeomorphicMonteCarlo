"""Current production consumes the equally calibrated final maps without changes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dcs_bench import reported_study as study
from scripts import reproduce_scs_100d as runner


class MatchedPreparationTests(unittest.TestCase):
    def test_final_maps_and_steps_match_both_pilot_winners_in_every_backend(self):
        asset = study._load_matched_calibration()
        for dimension in (100, 500):
            case = asset['dimensions'][str(dimension)]
            for backend in ('jax', 'numba'):
                config = study.configuration(comparison=backend, target=f'skew-t-{dimension}d')
                prepared = study.load_prepared(config)
                maps = runner.effective_method_maps(config, prepared['maps'])
                original = prepared['provenance']['original_fitted_maps']['sphere']
                self.assertEqual(maps['SCS'], original)
                self.assertEqual(runner.method_execution_config(config, 'SCS')['initial_scale'], .1)
                for method in config['algorithms']:
                    if method == 'SCS':
                        continue
                    family = 'SCS' if method == 'SCS-calibrated' else 'DCS-BW'
                    chosen = case['selections'][family]
                    mapping = maps[method]
                    self.assertEqual(mapping['scale'], chosen['R'])
                    self.assertEqual(mapping['mu'], [0.] * dimension)
                    self.assertEqual(mapping['observer' if family == 'SCS' else 'delta'], [0.] * dimension)
                    step = config['scs_calibrated_move'] if family == 'SCS' else config['dcs_fixed_move']
                    self.assertEqual(step, chosen['h'])
                self.assertFalse(prepared['provenance']['production_map_change_after_pilot'])
                self.assertTrue(prepared['provenance']['production_maps_calibrated_before_sampling'])
                self.assertTrue(config['calibrated_scs_map_intervention']['radius_step_recalibrated'])
                self.assertEqual(config['preparation_seed'], asset['seed'])

    def test_current_selection_is_highest_pilot_score_under_equal_resources(self):
        asset = study._load_matched_calibration()
        self.assertEqual(asset['final_map_policy'], {'DCS-BW': 'zero_mu_zero_delta', 'SCS': 'zero_mu_zero_observer'})
        for case in asset['dimensions'].values():
            self.assertEqual(case['pilot_transitions_per_family'], {'DCS-BW': 524288, 'SCS': 524288})
            for method in ('DCS-BW', 'SCS'):
                candidates = [r for r in case['candidates'] if r['method'] == method]
                self.assertEqual(len(candidates), 16)
                self.assertEqual(sum(r['transitions'] for r in candidates), 524288)
                selected = case['selections'][method]
                winner = next(r for r in candidates if r['id'] == selected['candidate_id'])
                self.assertEqual(winner['score'], max(r['score'] for r in candidates))
                self.assertEqual((winner['R'], winner['h']), (selected['R'], selected['h']))

    def test_stale_calibration_and_seed_cannot_be_mislabelled_as_current(self):
        config = study.configuration()
        for field, value in (
            ('reported_calibration_asset_sha256', study.CALIBRATION_SHA256),
            ('reported_calibration_asset', 'studies/zero_centre/maps/joint_calibration.json'),
            ('calibrated_scs_map_intervention', study._zero_scs_intervention()),
            ('preparation_seed', 123),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                study.validate_configuration(dict(config, **{field: value}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'matched_calibration.json').write_bytes((study.ASSETS/'joint_calibration.json').read_bytes())
            with patch.object(study, 'ASSETS', root), self.assertRaisesRegex(ValueError, 'Matched calibration hash'):
                study._load_matched_calibration()

    def test_legacy_zero_map_settings_remain_distinct_readable_inputs(self):
        old = study.zero_configuration(target='skew-t-500d')
        current = study.configuration(target='skew-t-500d')
        self.assertNotEqual(old['protocol'], current['protocol'])
        self.assertEqual(old['scs_calibrated_move'], .025)
        self.assertFalse(old['calibrated_scs_map_intervention']['radius_step_recalibrated'])
        self.assertTrue(current['calibrated_scs_map_intervention']['radius_step_recalibrated'])
        self.assertEqual(study.load_prepared(old)['maps']['sphere'], study.load_prepared(current)['maps']['sphere'])


if __name__ == '__main__':
    unittest.main()
