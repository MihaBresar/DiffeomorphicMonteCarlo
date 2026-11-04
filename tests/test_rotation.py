"""Target-preserving angular proposals and equal-budget two-scale calibration."""
import unittest

import numpy as np

from dcs_bench import rotation
from dcs_bench.references import skew_t_iid
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, mobius
from dcs_bench.timed_radial import _dcs_components


class RotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        jax.config.update('jax_enable_x64', True)

    def components(self, mapping, nu=3., alpha=None):
        target = Target(dict(kind='skew_t', dimension=mapping.d, nu=nu,
                             alpha={'0': 4., '1': -4.} if alpha is None else alpha))
        reflected = _dcs_components(mapping.parameters(), target, 'DCS-BW-reflect')
        return target, rotation.build_components(mapping.parameters(), target, reflected)

    def test_rotation_preserves_centered_physical_radius_and_reverses_with_vi(self):
        for beta in (1., 2.):
            for delta in (np.zeros(5), np.array([.2, -.1, .03, 0., 0.])):
                mapping = BallMap(5, beta=beta, scale=1.8,
                                  mu=np.array([.4, -.2, .1, .05, -.1]), delta=delta)
                w = np.array([.35, -.2, .3, -.1, .25])
                state = mobius(w, -delta)[0]
                normal = np.array([.6, .4, -.1, .2, .3])
                angular_scale = .7
                proposed, correction, valid = rotation.angular_proposal(state, normal, angular_scale, delta)
                proposed = np.asarray(proposed)
                with self.subTest(beta=beta, delta=delta):
                    self.assertTrue(bool(valid))
                    x, _ = mapping.forward(state)
                    y, _ = mapping.forward(proposed)
                    self.assertAlmostEqual(np.linalg.norm(x-mapping.mu), np.linalg.norm(y-mapping.mu), places=12)
                    u = w/np.linalg.norm(w)
                    v = normal-u*np.dot(u, normal)
                    v /= np.linalg.norm(v)
                    theta = 2.*np.arctan(angular_scale)
                    reverse_tangent = np.sin(theta)*u-np.cos(theta)*v
                    reverse, reverse_correction, reverse_valid = rotation.angular_proposal(
                        proposed, reverse_tangent, angular_scale, delta)
                    self.assertTrue(bool(reverse_valid))
                    np.testing.assert_allclose(reverse, state, rtol=1e-12, atol=1e-12)
                    self.assertAlmostEqual(float(correction+reverse_correction), 0., places=12)

    def test_mobius_proposal_correction_equals_physical_target_ratio(self):
        mapping = BallMap(5, scale=1.8, mu=[.4, -.2, .1, .05, -.1],
                          delta=[.2, -.1, .03, 0., 0.])
        target, components = self.components(mapping)
        state = np.array([.35, -.2, .3, -.1, .25])
        proposed, correction, valid = rotation.angular_proposal(
            state, np.array([.6, .4, -.1, .2, .3]), .7, mapping.delta)
        self.assertTrue(bool(valid))
        physical_ratio = target.log_prob(mapping.forward(np.asarray(proposed))[0])-target.log_prob(mapping.forward(state)[0])
        raw_ratio = float(components['density'](proposed)-components['density'](state))
        self.assertGreater(abs(float(correction)), 1e-4)
        self.assertAlmostEqual(raw_ratio+float(correction), physical_ratio, places=11)
        self.assertGreater(abs(raw_ratio-physical_ratio), 1e-4)

    def test_degenerate_proposals_self_loop_without_clipping(self):
        delta = np.zeros(5)
        for state, normal, scale in ((np.zeros(5), np.ones(5), 1.),
                                     (np.full(5, .1), np.zeros(5), 1.),
                                     (np.full(5, .1), np.ones(5), 0.),
                                     (np.full(5, .1), np.ones(5), np.nan)):
            proposed, correction, valid = rotation.angular_proposal(state, normal, scale, delta)
            self.assertFalse(bool(valid))
            np.testing.assert_array_equal(proposed, state)
            self.assertEqual(float(correction), 0.)

    def test_symmetric_angular_kernel_accepts_all_and_counts_every_state(self):
        import jax
        import jax.numpy as jnp
        mapping = BallMap(7, scale=np.sqrt(3.))
        _, components = self.components(mapping, alpha={})
        initial = jnp.asarray([.45, -.2, .1, .1, 0., 0., 0.])
        states, accepted = jax.jit(lambda key: components['angular_block'](
            initial, key, 600, 1.))(jax.random.key(282))
        self.assertEqual(states.shape, (600, 7))
        np.testing.assert_array_equal(accepted, 1.)
        np.testing.assert_allclose(np.linalg.norm(states, axis=1), np.linalg.norm(initial), rtol=1e-12, atol=1e-12)
        self.assertGreater(np.mean(np.sum(np.diff(np.asarray(states), axis=0)**2, axis=1)), .1)

    def test_mixture_makes_one_fair_component_choice_per_transition(self):
        import jax
        import jax.numpy as jnp
        # Under uniform ball density both components accept. Only rotations
        # preserve radius exactly, so radial differences reveal branch choices.
        mapping = BallMap(7, scale=np.sqrt(2.))
        _, components = self.components(mapping, nu=2., alpha={})
        initial = jnp.asarray([.45, -.2, .1, .1, 0., 0., 0.])
        states, accepted = jax.jit(lambda key: components['block'](
            initial, key, 1600, jnp.array([.4, 1.])))(jax.random.key(8231))
        self.assertEqual(states.shape, (1600, 7))
        np.testing.assert_array_equal(accepted, 1.)
        radii = np.linalg.norm(np.vstack((initial, states)), axis=1)
        angular_fraction = np.mean(np.abs(np.diff(radii)) < 1e-12)
        self.assertGreater(angular_fraction, .43)
        self.assertLess(angular_fraction, .57)
        self.assertGreater(np.ptp(radii), .1)

    def test_vi_on_angular_kernel_preserves_skew_target_from_independent_starts(self):
        import jax
        import jax.numpy as jnp
        mapping = BallMap(5, scale=1.4, mu=[.15, -.08, .03, 0., 0.],
                          delta=[.12, -.07, .02, 0., 0.])
        target, components = self.components(mapping)
        physical = skew_t_iid(target, np.random.default_rng(9137), 6000)
        starts = mapping.inverse(physical)
        keys = jax.random.split(jax.random.key(2923), len(starts))
        def step(state, key):
            states, _ = components['angular_block'](state, key, 1, .8)
            return components['projection'](states)[0]
        outputs = np.asarray(jax.jit(jax.vmap(step))(jnp.asarray(starts), keys))
        np.testing.assert_allclose(np.linalg.norm(outputs-mapping.mu, axis=1),
                                   np.linalg.norm(physical-mapping.mu, axis=1), rtol=5e-10, atol=1e-10)
        # Paired bounded-function changes have mean zero under invariance.
        difference = np.tanh(outputs)-np.tanh(physical)
        se = difference.std(axis=0, ddof=1)/np.sqrt(len(difference))
        np.testing.assert_array_less(np.abs(difference.mean(axis=0)), 6.*se+1e-12)

    def test_split_calibration_accounts_for_all_pilots_and_frozen_scales(self):
        import jax
        import jax.numpy as jnp
        def advance(state, key, count, scale):
            return state+scale*jnp.arange(1, count+1)[:, None], jnp.ones(count)
        components = dict(reflect_block=advance, angular_block=advance, direction=lambda x:x)
        for warmup in (6, 7, 100):
            state, scales, accepts, histories = jax.jit(lambda key: rotation.calibrate(
                components, jnp.array([0.]), key, warmup, .1, 1.))(jax.random.key(77))
            bw, angle = map(np.asarray, histories)
            with self.subTest(warmup=warmup):
                self.assertEqual(int(bw[:, :, 1].sum()), warmup//2)
                self.assertEqual(int(angle[:, :, 1].sum()), warmup-warmup//2)
                self.assertEqual(float(accepts), warmup)
                expected = np.sum(bw[:, :, 0]*bw[:, :, 1])+np.sum(angle[:, :, 0]*angle[:, :, 1])
                self.assertAlmostEqual(float(state[0]), expected, places=9)
                for scale, history in zip(scales, (bw, angle)):
                    selected = history[-1, np.argmax(history[-1, :, 4]), 0]
                    self.assertAlmostEqual(float(scale), selected, places=10)

    def test_invalid_warmup_and_scales_fail_before_kernel_execution(self):
        def unused(*args):
            self.fail('Invalid calibration settings must fail before sampling')
        components = dict(reflect_block=unused, angular_block=unused, direction=unused)
        for warmup in (0, 5, 6.5, True):
            with self.assertRaises(ValueError):
                rotation.calibrate(components, [0.], None, warmup, .1)
        for bw, angular in ((0., 1.), (.1, 0.), (.1, np.nan), (True, 1.), (.1, True)):
            with self.assertRaises(ValueError):
                rotation.calibrate(components, [0.], None, 100, bw, angular)


if __name__ == '__main__':
    unittest.main()
