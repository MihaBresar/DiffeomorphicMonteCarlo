"""Actual-kernel, exact-budget and robust-observable pilot checks."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from dcs_bench import timed_radial as timed
from dcs_bench.radial_tuning import (
    calibrate_radial, log1p_norm, radial_jump_scores, COARSE_FACTORS,
    REFINE_FACTORS, HISTORY_COLUMNS, WINDOWS, POLICY)
from dcs_bench.transforms import BallMap, SphereMap


class RadialTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        jax.config.update('jax_enable_x64', True)

    def test_invalid_settings_reject_before_invoking_the_kernel(self):
        def unused(*args):
            self.fail('Invalid calibration must not invoke the kernel')
        for warmup in (0, 3, 7, -1, 8.5, True):
            with self.subTest(warmup=warmup), self.assertRaises(ValueError):
                calibrate_radial(unused, unused, [0.], None, warmup, .1)
        for scale in (0., -1., np.inf, np.nan, True):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                calibrate_radial(unused, unused, [0.], None, 100, scale)
        for ceiling in (.01, np.inf, np.nan, False):
            with self.subTest(ceiling=ceiling), self.assertRaises(ValueError):
                calibrate_radial(unused, unused, [0.], None, 100, .1, ceiling)

    def test_physical_radial_score_ignores_angle_and_handles_extreme_finite_norms(self):
        import jax.numpy as jnp
        points = jnp.array([[0., 0.], [3., 4.], [-3., -4.], [3e300, 4e300], [3e300, 4e300]])
        observed = np.asarray(log1p_norm(points))
        expected = np.array([0., np.log(6.), np.log(6.), np.log(5.)+300*np.log(10.),
                             np.log(5.)+300*np.log(10.)])
        np.testing.assert_allclose(observed, expected, rtol=2e-15, atol=1e-14)
        delta = np.diff(expected)
        np.testing.assert_allclose(radial_jump_scores(points), delta**2/(1+delta**2), rtol=1e-14)
        self.assertTrue(np.all(np.isfinite(observed)))
        self.assertEqual(float(radial_jump_scores(points)[1]), 0.)

    def test_budget_state_continuity_randomized_order_and_replay(self):
        import jax
        import jax.numpy as jnp
        def advance(state, key, count, scale):
            return state + scale*jnp.arange(1, count+1)[:, None], jnp.ones(count)
        for warmup in (8, 9, 15, 100, 1000, 1500):
            with self.subTest(warmup=warmup):
                program = jax.jit(lambda key: calibrate_radial(
                    advance, lambda states: states, jnp.array([0.]), key, warmup, .1))
                result = program(jax.random.key(72)); repeated = program(jax.random.key(72))
                for actual, replay in zip(result, repeated):
                    np.testing.assert_array_equal(actual, replay)
                state, scale, accepts, history = result
                history = np.asarray(history)
                self.assertEqual(history.shape, (WINDOWS, len(HISTORY_COLUMNS)))
                np.testing.assert_array_equal(np.sort(history[:, 5]), np.arange(WINDOWS))
                np.testing.assert_allclose(history[:5, 0], .1*np.array(COARSE_FACTORS), rtol=1e-14)
                winner = history[np.argmax(history[:5, 4]), 0]
                np.testing.assert_allclose(history[5:, 0], winner*np.array(REFINE_FACTORS), rtol=1e-14)
                base, extra = divmod(warmup, WINDOWS)
                chronological = history[np.argsort(history[:, 5])]
                np.testing.assert_array_equal(chronological[:, 1], base+(np.arange(WINDOWS)<extra))
                self.assertEqual(int(history[:, 1].sum()), warmup)
                self.assertEqual(float(accepts), warmup)
                self.assertAlmostEqual(float(state[0]), float(np.sum(history[:, 0]*history[:, 1])), places=8)
                self.assertAlmostEqual(float(scale), history[5+np.argmax(history[5:, 4]), 0])
                self.assertEqual(history[:5, 4].sum(), 1.)
                self.assertEqual(history[5:, 4].sum(), 1.)
                self.assertFalse(np.array_equal(history[:, 5], np.arange(WINDOWS)))

    def test_first_move_and_rejected_repeats_are_scored_without_filtering(self):
        import jax
        import jax.numpy as jnp
        def first_move_only(state, key, count, scale):
            return jnp.repeat((state+.25)[None, :], count, axis=0), \
                jnp.where(jnp.arange(count)==0, 1., 0.)
        state, _, accepts, history = jax.jit(lambda key: calibrate_radial(
            first_move_only, lambda states: states, jnp.array([0.]), key, 1000, .1))(jax.random.key(8))
        history = np.asarray(history)
        chronological = history[np.argsort(history[:, 5])]
        previous = .25*np.arange(WINDOWS)
        delta = np.log1p(previous+.25)-np.log1p(previous)
        np.testing.assert_allclose(chronological[:, 2]*chronological[:, 1], delta**2/(1+delta**2), rtol=1e-13)
        np.testing.assert_allclose(history[:, 3]*history[:, 1], 1., rtol=1e-14)
        self.assertEqual(float(accepts), WINDOWS)
        self.assertEqual(float(state[0]), WINDOWS*.25)

    def test_angular_motion_at_full_acceptance_does_not_force_scale_increase(self):
        import jax
        import jax.numpy as jnp
        def rotate(state, key, count, scale):
            return state*jnp.where(jnp.arange(count)%2==0, -1., 1.)[:, None], jnp.ones(count)
        _, scale, accepts, history = jax.jit(lambda key: calibrate_radial(
            rotate, lambda states: states, jnp.array([1., 0.]), key, 1000, .1))(jax.random.key(9))
        np.testing.assert_array_equal(np.asarray(history)[:, 2], 0.)
        self.assertAlmostEqual(float(scale), .1/8.)
        self.assertEqual(float(accepts), 1000.)

    def test_refines_actual_radial_optimum_and_retains_broad_search(self):
        import jax
        import jax.numpy as jnp
        def peaked(state, key, count, scale):
            delta = .02/jnp.sqrt(1.+jnp.log2(scale/.4)**2)
            return state+delta*jnp.arange(1, count+1)[:, None], jnp.ones(count)
        _, scale, _, history = jax.jit(lambda key: calibrate_radial(
            peaked, jnp.expm1, jnp.array([0.]), key, 1500, .1))(jax.random.key(19))
        history = np.asarray(history)
        self.assertAlmostEqual(float(scale), .4)
        self.assertAlmostEqual(history[:5, 0].max(), 6.4)
        self.assertGreater(history[:, 0].max(), 2.)
        np.testing.assert_allclose(history[5:, 0], [.2, .4, .8], rtol=1e-14)
        expected_delta = .02/np.sqrt(1.+np.log2(history[:, 0]/.4)**2)
        np.testing.assert_allclose(history[:, 2], expected_delta**2/(1+expected_delta**2), rtol=1e-12)

    def test_new_policy_uses_only_actual_kernel_and_legacy_path_is_preserved(self):
        import jax
        import jax.numpy as jnp
        calls = []
        def components(mapping, spec, algorithm):
            calls.append(algorithm)
            def block(state, key, count, scale):
                return jnp.repeat(state[None, :], count, axis=0), jnp.zeros(count)
            return block, lambda states: states, jnp.ones(2), lambda state: jnp.asarray(0.)
        for algorithm in (timed.BILLIARD_METHOD, 'SCS'):
            with self.subTest(algorithm=algorithm), patch.object(timed, '_components', side_effect=components):
                calls.clear(); timed._programs.cache_clear()
                burn, batch, initial, _ = timed._programs('{}', '{}', algorithm, 100, 1, 50., .1,
                    tuning_policy=POLICY)
                _, scale, accepts, history = burn(initial, jax.random.key(22))
                batch(initial, jax.random.key(23), scale)
                self.assertEqual(calls, [algorithm])
                self.assertEqual(float(accepts), 0.)
                metadata = timed._radial_tuning_metadata(np.asarray(history), float(scale), 100, algorithm)
                self.assertEqual(metadata['calibration_algorithm'], algorithm)
                self.assertEqual(metadata['tuning_pilot_iterations'], 100)
                self.assertEqual(len(metadata['radial_tuning_history']), 8)
                self.assertNotIn('tuning_history', metadata)
        # The old policy still invokes reflected BW, with no implicit migration.
        with patch.object(timed, '_components', side_effect=components):
            calls.clear(); timed._programs.cache_clear()
            timed._programs(json.dumps(BallMap(2).parameters()), '{}', timed.BILLIARD_METHOD,
                100, 1, 50., .1, tuning_policy=timed.BALL_TUNING)
            self.assertEqual(calls, [timed.BILLIARD_METHOD, 'DCS-BW-reflect'])
        timed._programs.cache_clear()

    def test_real_kernels_freeze_own_calibration_and_native_reuses_it(self):
        spec = dict(kind='skew_t', dimension=5, nu=3., alpha={'0':3., '1':-3.})
        for algorithm, mapping in ((timed.BILLIARD_METHOD, BallMap(5, scale=1.3,
                delta=np.array([.1, 0., 0., 0., 0.])).parameters()),
                ('SCS', SphereMap(5, scale=1.3).parameters())):
            with self.subTest(algorithm=algorithm):
                config = dict(algorithm=algorithm, target=spec, warmup=100, thin=1,
                    time_budget=.001, step_tuning_policies={algorithm:POLICY})
                original = json.dumps(mapping, sort_keys=True)
                first = timed._Matched(mapping, config, 229)
                repeat = timed._Matched(mapping, config, 229)
                np.testing.assert_array_equal(first.state, repeat.state)
                self.assertEqual(first.scale, repeat.scale)
                self.assertEqual(first.tuning_metadata, repeat.tuning_metadata)
                self.assertEqual(first.tuning_metadata['calibration_algorithm'], algorithm)
                self.assertEqual(sum(row['iterations'] for row in first.tuning_metadata['radial_tuning_history']), 100)
                self.assertEqual(json.dumps(mapping, sort_keys=True), original)
                frozen = first.scale
                prefixes, norms = first.batch()
                self.assertTrue(np.all(np.isfinite(prefixes)))
                self.assertEqual(len(norms), 300)
                self.assertEqual(first.scale, frozen)
                if algorithm == timed.BILLIARD_METHOD:
                    native = timed._backend(mapping, dict(config, dcs_backend='numba'), 229)
                    np.testing.assert_array_equal(native.state, repeat.state)
                    self.assertEqual(native.scale, repeat.scale)
                    self.assertEqual({key:native.tuning_metadata[key] for key in repeat.tuning_metadata},
                                     repeat.tuning_metadata)

    def test_explicit_ceiling_and_numerical_safeguards_do_not_change_policy_scope(self):
        import jax
        import jax.numpy as jnp
        def unchanged(state, key, count, scale):
            return jnp.repeat(state[None, :], count, axis=0), jnp.ones(count)
        for initial, ceiling in ((.1, 1.), (np.exp(14.), None)):
            limit = np.exp(15.) if ceiling is None else ceiling
            _, _, _, history = jax.jit(lambda key: calibrate_radial(unchanged,
                lambda states: states, jnp.array([1.]), key, 100, initial, ceiling))(jax.random.key(3))
            self.assertLessEqual(np.asarray(history)[:, 0].max(), limit*(1+1e-14))
            self.assertGreaterEqual(np.asarray(history)[:, 0].min(), np.exp(-30.)*(1-1e-14))
        for method in ('DCS-RWM', 'DCS-BW', 'DCS-BW-reflect'):
            with self.subTest(method=method), self.assertRaises(ValueError):
                timed.resolve_step_tuning(dict(algorithm=method, warmup=100,
                    step_tuning_policies={method:POLICY}))
        self.assertEqual(timed.resolve_step_tuning(dict(algorithm=timed.BILLIARD_METHOD,
            warmup=100)),timed.BALL_TUNING)
        self.assertEqual(timed.resolve_step_tuning(dict(algorithm='SCS',warmup=100)),timed.ACCEPTANCE_TUNING)


if __name__ == '__main__':
    unittest.main()
