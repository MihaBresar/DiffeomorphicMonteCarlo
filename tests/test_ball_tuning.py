"""Behavioral checks for discarded-pilot reflected-BW movement calibration."""
import unittest

import numpy as np

from dcs_bench.ball_tuning import calibrate_ball_esjd, HISTORY_COLUMNS, MAX_ROUNDS


class BallTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        jax.config.update('jax_enable_x64', True)

    def test_invalid_settings_fail_before_invoking_the_kernel(self):
        def unused(*args):
            self.fail('Kernel must not run for invalid settings')

        for warmup in (0, 1, 2, -1, 3.5, True):
            with self.subTest(warmup=warmup), self.assertRaises(ValueError):
                calibrate_ball_esjd(unused, [0.], None, warmup, .1)
        for initial in (0., -1., np.inf, np.nan, True):
            with self.subTest(initial=initial), self.assertRaises(ValueError):
                calibrate_ball_esjd(unused, [0.], None, 100, initial)
        for ceiling in (.01, np.inf, np.nan, False):
            with self.subTest(ceiling=ceiling), self.assertRaises(ValueError):
                calibrate_ball_esjd(unused, [0.], None, 100, .1, ceiling)

    def test_zero_movement_at_full_acceptance_does_not_drive_scale_up(self):
        import jax
        import jax.numpy as jnp

        def stationary(state, key, count, scale):
            return jnp.repeat(state[None, :], count, axis=0), jnp.ones(count)

        initial = jnp.array([.2, -.1])
        state, scale, accepts, history = jax.jit(lambda key: calibrate_ball_esjd(
            stationary, initial, key, 100, .1))(jax.random.key(1))
        np.testing.assert_array_equal(state, initial)
        self.assertEqual(float(accepts), 100.)
        self.assertAlmostEqual(float(scale), .1 / 2 ** MAX_ROUNDS)
        np.testing.assert_array_equal(np.asarray(history)[:, :, 2], 0.)
        np.testing.assert_array_equal(np.asarray(history)[:, :, 3], 1.)
        np.testing.assert_array_equal(np.asarray(history)[:, :, 4], np.tile([1., 0., 0.], (MAX_ROUNDS, 1)))

    def test_pilot_can_select_a_useful_scale_above_two(self):
        import jax
        import jax.numpy as jnp

        # A synthetic accepted-state path has maximum movement at h=3.2.
        # Every state stays inside the unit interval and every step accepts;
        # the objective, rather than an acceptance target or cap2, must win.
        def peaked(state, key, count, scale):
            amplitude = .4 / jnp.sqrt(1. + (jnp.log2(scale / 3.2)) ** 2)
            sign = jnp.where(state[0] >= 0., 1., -1.)
            positions = sign * jnp.where(jnp.arange(count) % 2 == 0, -amplitude, amplitude)
            return positions[:, None], jnp.ones(count)

        state, scale, accepts, history = jax.jit(lambda key: calibrate_ball_esjd(
            peaked, jnp.array([0.]), key, 2400, .1))(jax.random.key(7))
        self.assertAlmostEqual(float(scale), 3.2)
        self.assertGreater(float(scale), 2.)
        self.assertEqual(float(accepts), 2400.)
        self.assertLess(abs(float(state[0])), 1.)
        self.assertGreater(np.asarray(history)[:, :, 0].max(), 3.2)
        self.assertGreater(np.asarray(history)[-1, 1, 2], 0.)

    def test_budget_rotation_and_state_continuity_include_every_trial(self):
        import jax
        import jax.numpy as jnp

        # A deterministic recording kernel: each transition adds its scale.
        # Its state is a bookkeeping counter, not a target-distribution test.
        def advance(state, key, count, scale):
            return state + scale * jnp.arange(1, count + 1)[:, None], jnp.ones(count)

        for warmup in (3, 4, 7, 23, 24, 25, 100, 1000, 1500):
            with self.subTest(warmup=warmup):
                state, scale, accepts, history = jax.jit(lambda key: calibrate_ball_esjd(
                    advance, jnp.array([0.]), key, warmup, .1))(jax.random.key(2))
                history = np.asarray(history)
                rounds = min(MAX_ROUNDS, warmup // 3)
                self.assertEqual(history.shape, (rounds, 3, len(HISTORY_COLUMNS)))
                self.assertEqual(int(history[:, :, 1].sum()), warmup)
                self.assertEqual(float(accepts), warmup)
                base, extra = divmod(warmup, rounds * 3)
                chronological = 0
                expected_endpoint = 0.
                for round_index, rows in enumerate(history):
                    np.testing.assert_array_less(rows[:-1, 0], rows[1:, 0])
                    for trial_index in range(3):
                        row = rows[(trial_index + round_index) % 3]
                        count = base + (chronological < extra)
                        self.assertEqual(int(row[1]), count)
                        expected_endpoint += count * row[0]
                        chronological += 1
                    np.testing.assert_allclose(rows[:, 2], rows[:, 0] ** 2, rtol=1e-12)
                    np.testing.assert_array_equal(rows[:, 4], [0., 0., 1.])
                self.assertAlmostEqual(float(state[0]), expected_endpoint, places=9)
                self.assertAlmostEqual(float(scale), history[-1, 2, 0])

    def test_first_transition_and_rejected_repeats_enter_window_scores(self):
        import jax
        import jax.numpy as jnp

        def first_move_only(state, key, count, scale):
            return jnp.repeat((state + .25)[None, :], count, axis=0), \
                jnp.where(jnp.arange(count) == 0, 1., 0.)

        state, _, accepts, history = jax.jit(lambda key: calibrate_ball_esjd(
            first_move_only, jnp.array([0.]), key, 100, .1))(jax.random.key(4))
        history = np.asarray(history)
        counts = history[:, :, 1]
        np.testing.assert_allclose(history[:, :, 2] * counts, .25 ** 2, rtol=1e-14)
        np.testing.assert_allclose(history[:, :, 3] * counts, 1., rtol=1e-14)
        self.assertEqual(float(accepts), 24.)
        self.assertEqual(float(state[0]), 24. * .25)

    def test_numerical_limits_and_explicit_historical_limit_are_respected(self):
        import jax
        import jax.numpy as jnp

        def advance(state, key, count, scale):
            return state + scale * jnp.arange(1, count + 1)[:, None], jnp.ones(count)

        for initial, limit, expected in ((np.exp(14.), None, np.exp(15.)), (.1, 1., 1.)):
            with self.subTest(limit=limit):
                _, scale, _, history = jax.jit(lambda key: calibrate_ball_esjd(
                    advance, jnp.array([0.]), key, 48, initial, limit))(jax.random.key(9))
                self.assertAlmostEqual(float(scale) / expected, 1.)
                self.assertLessEqual(float(np.asarray(history)[:, :, 0].max()), expected * (1. + 1e-14))

    def test_uniform_transformed_target_accepts_all_without_tuning_runaway(self):
        import jax
        import jax.numpy as jnp
        from dcs_bench.targets import Target
        from dcs_bench.transforms import BallMap
        from dcs_bench.timed_radial import _dcs_components

        # t(2), scale=sqrt(2), beta=2 maps exactly to uniform ball volume.
        target = Target({'kind': 'skew_t', 'dimension': 5, 'nu': 2., 'alpha': {}})
        mapping = BallMap(5, beta=2., scale=np.sqrt(2.))
        block, _, initial, density = _dcs_components(mapping.parameters(), target, 'DCS-BW-reflect')
        points = jnp.array([[0., 0., 0., 0., 0.], [.8, .1, 0., 0., 0.],
                            [.2, -.3, .5, 0., 0.]])
        logdensities = np.asarray(jax.vmap(density)(points))
        np.testing.assert_allclose(logdensities, logdensities[0], atol=1e-12, rtol=0)
        calibrate = jax.jit(lambda key: calibrate_ball_esjd(block, initial, key, 240, .1))
        result = jax.block_until_ready(calibrate(jax.random.key(181)))
        repeat = jax.block_until_ready(calibrate(jax.random.key(181)))
        for first, second in zip(result, repeat):
            np.testing.assert_array_equal(first, second)
        state, scale, accepts, history = result
        self.assertEqual(float(accepts), 240.)
        self.assertTrue(np.all(np.isfinite(state)))
        self.assertLess(np.linalg.norm(state), 1.)
        self.assertGreater(float(scale), 0.)
        self.assertLess(float(scale), np.exp(15.))
        self.assertGreater(np.asarray(history)[:, :, 2].max(), .01)
        states, accepted = jax.jit(lambda state: block(state, jax.random.key(182), 200, scale))(state)
        self.assertTrue(np.all(np.isfinite(states)))
        self.assertTrue(np.all(np.linalg.norm(states, axis=1) < 1.))
        np.testing.assert_array_equal(accepted, 1.)


if __name__ == '__main__':
    unittest.main()
