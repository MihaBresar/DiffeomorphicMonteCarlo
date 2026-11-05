"""Prespecified DCS moves preserve kernels, discarded warmup and all-state estimates."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from dcs_bench import timed_radial as timed
from dcs_bench.transforms import BallMap, SphereMap

METHODS = ('DCS-RWM', 'DCS-BW', 'DCS-BW-reflect', 'DCS-Billiard')


def configuration(method='DCS-BW', dimension=100, radius=.07, backend='jax'):
    return dict(algorithm=method, dimension=dimension,
        target={'kind': 'skew_t', 'dimension': dimension, 'nu': 3., 'alpha': {'0': 5.}},
        warmup=16, thin=1, time_budget=.001, threshold=50., initial_scale=.1,
        ess_enabled=False, coordinate_metrics=False, dcs_backend=backend,
        dcs_fixed_move=radius,
        step_tuning_policies={**{method: timed.FIXED_DCS_TUNING for method in METHODS},
            'SCS': timed.SCS_FIXED_TUNING})


class FixedDCSStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        jax.config.update('jax_enable_x64', True)

    def test_fixed_move_matches_uniform_ball_coordinate_variance(self):
        for dimension in (100, 500):
            for radius in (.07, .15):
                gaussian = timed._initial_proposal_scale(configuration('DCS-RWM', dimension, radius))
                for method in METHODS[1:]:
                    ball = timed._initial_proposal_scale(configuration(method, dimension, radius))
                    self.assertEqual(ball, radius)
                    self.assertAlmostEqual(gaussian**2, ball**2/(dimension+2), places=16)

    def test_invalid_policies_moves_and_warmup_are_rejected(self):
        for radius in (None, 0., -.1, np.inf, np.nan, True, '0.07'):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                timed.resolve_step_tuning(configuration(radius=radius))
        for method in ('SCS', timed.ROTATION_METHOD, timed.RWM_ROTATION_METHOD, 'unknown'):
            config = configuration(method)
            config['step_tuning_policies'] = {method: timed.FIXED_DCS_TUNING}
            with self.subTest(method=method), self.assertRaises(ValueError):
                timed.resolve_step_tuning(config)
        for warmup in (0, -1, True, 1.5):
            config = configuration()
            config['warmup'] = warmup
            with self.subTest(warmup=warmup), self.assertRaises(ValueError):
                timed.resolve_step_tuning(config)

    def test_scale_ceiling_is_checked_against_actual_normalized_step(self):
        config = configuration('DCS-RWM', radius=.15)
        config['step_scale_ceilings'] = {'DCS-RWM': .02}
        self.assertEqual(timed.resolve_step_scale_ceiling(config), .02)
        config['step_scale_ceilings']['DCS-RWM'] = .01
        with self.assertRaises(ValueError):
            timed.resolve_step_scale_ceiling(config)
        config = configuration('DCS-BW', radius=.15)
        config['step_scale_ceilings'] = {'DCS-BW': .1}
        with self.assertRaises(ValueError):
            timed.resolve_step_scale_ceiling(config)

    def test_discarded_phase_exactly_matches_the_unchanged_production_kernel(self):
        import jax
        import jax.numpy as jnp
        seed = 138
        for method in METHODS:
            config = configuration(method, radius=.15)
            mapping = BallMap(100, scale=2., mu=[.1]+[0.]*99, delta=[.12]+[0.]*99).parameters()
            original = deepcopy(mapping)
            block, _, initial, _ = timed._components(mapping, config['target'], method)
            burn_key, production_key = jax.random.split(jax.random.key(seed))
            scale = .15 / np.sqrt(102.) if method == 'DCS-RWM' else .15
            direct = jax.jit(lambda state, key: block(state, key, 16, jnp.asarray(scale)))
            states, accepted = direct(initial, burn_key)
            with patch.object(timed, '_calibrate', side_effect=AssertionError('No adaptation allowed')), \
                    patch('dcs_bench.radial_tuning.calibrate_radial', side_effect=AssertionError('No radial pilot')), \
                    patch('dcs_bench.ball_tuning.calibrate_ball_esjd', side_effect=AssertionError('No ball pilot')):
                actual = timed._Matched(mapping, config, seed)
            np.testing.assert_array_equal(actual.state, states[-1])
            np.testing.assert_array_equal(jax.random.key_data(actual.key), jax.random.key_data(production_key))
            self.assertEqual(actual.warmup_acceptance, float(np.sum(accepted))/16)
            self.assertEqual(actual.scale, scale)
            self.assertEqual(actual.tuning_metadata['calibration_algorithm'], method)
            self.assertNotIn('radial_tuning_history', actual.tuning_metadata)
            self.assertNotIn('tuning_history', actual.tuning_metadata)
            self.assertEqual(mapping, original)
            if method == 'DCS-Billiard':
                effective = actual.tuning_metadata['effective_map']
                self.assertEqual(effective['delta'], [0.]*100)
                for field in ('mu', 'scale', 'beta'):
                    self.assertEqual(effective[field], mapping[field])
                self.assertIn('fixed-step own-kernel warmup', actual.implementation)
                self.assertIn('Fixed-step own billiard warmup', actual.tuning_metadata['calibration_coordinates'])
                self.assertNotIn('radial-movement pilot', actual.tuning_metadata['calibration_coordinates'])

    def test_timed_100d_and_500d_use_fixed_steps_and_include_rejection_repeats(self):
        for dimension in (100, 500):
            for radius in (.07, .15):
                for method in METHODS:
                    with self.subTest(dimension=dimension, radius=radius, method=method):
                        config = configuration(method, dimension, radius)
                        with patch.object(timed, '_calibrate', side_effect=AssertionError('No adaptation allowed')):
                            traces, record = timed.run_timed(BallMap(dimension, scale=2.).parameters(), config, 138)
                        scale = radius/np.sqrt(dimension+2.) if method == 'DCS-RWM' else radius
                        self.assertEqual(record['tuning_policy'], timed.FIXED_DCS_TUNING)
                        for name in ('initial_scale', 'scale', 'final_scale', 'frozen_scale', 'fixed_step'):
                            self.assertEqual(record[name], scale)
                        self.assertEqual(record['fixed_move_radius'], radius)
                        self.assertEqual(record['calibration_algorithm'], method)
                        self.assertNotIn('radial_tuning_history', record)
                        self.assertNotIn('tuning_history', record)
                        self.assertFalse(record['step_size_adapted'])
                        self.assertFalse(record['adaptation_during_sampling'])
                        self.assertIsNone(record['target_acceptance'])
                        self.assertEqual(record['warmup_iterations'], 16)
                        self.assertEqual(record['tuning_pilot_iterations'], 16)
                        self.assertEqual(record['discarded_iterations'], record['iterations']//3)
                        self.assertEqual(len(traces), record['estimation_iterations'])
                        self.assertLess(record['accepted'], record['iterations'])
                        self.assertTrue(np.any(np.diff(traces[:, 1]) == 0.))
                        np.testing.assert_allclose(np.mean(traces, axis=0), record['mean_all_iterations'],
                                                   rtol=2e-13, atol=1e-13)

    def test_numba_receives_identical_fixed_step_and_discarded_endpoint(self):
        mapping = BallMap(100, scale=2., delta=[.12]+[0.]*99).parameters()
        for method in METHODS:
            with self.subTest(method=method):
                config = configuration(method, radius=.15)
                jax_driver = timed._backend(mapping, config, 138)
                native = timed._backend(mapping, dict(config, dcs_backend='numba'), 138)
                np.testing.assert_array_equal(native.state, jax_driver.state)
                self.assertEqual(native.scale, jax_driver.scale)
                self.assertEqual(native.warmup_acceptance, jax_driver.warmup_acceptance)
                for key in ('calibrated_state_sha256', 'fixed_step', 'fixed_move_radius', 'step_size_adapted',
                            'calibration_algorithm'):
                    self.assertEqual(native.tuning_metadata[key], jax_driver.tuning_metadata[key])
                self.assertNotIn('radial_tuning_history', native.tuning_metadata)
                self.assertNotIn('tuning_history', native.tuning_metadata)

    def test_scs_fixed_point_one_is_unchanged_by_other_methods_fixed_controls(self):
        with_controls = configuration('SCS')
        original = deepcopy(with_controls)
        original.pop('dcs_fixed_move')
        original['step_tuning_policies'] = {'SCS': timed.SCS_FIXED_TUNING}
        mapping = SphereMap(100, scale=2.).parameters()
        expected = timed._Matched(mapping, original, 138)
        actual = timed._Matched(mapping, with_controls, 138)
        self.assertEqual(actual.scale, .1)
        self.assertEqual(actual.tuning_policy, timed.SCS_FIXED_TUNING)
        np.testing.assert_array_equal(actual.state, expected.state)
        self.assertEqual(actual.tuning_metadata, expected.tuning_metadata)
        self.assertEqual(actual.warmup_acceptance, expected.warmup_acceptance)


if __name__ == '__main__':
    unittest.main()
