"""All-state coordinate statistics for optimized DCS on the original targets."""
from types import SimpleNamespace
import unittest

import numpy as np
from scipy import special

from dcs_bench import numba_radial as native
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap
from tests.test_numba_radial import reflected_reference


def target_spec(d=6):
    return dict(kind='skew_t', dimension=d, nu=3., alpha={'0': 8., '1': -3.})


def direct_log_prob(spec, x):
    """Physical skew-t density independent of the native cancellation formula."""
    d, nu = spec['dimension'], spec['nu']
    alpha = np.zeros(d)
    for key, value in spec['alpha'].items():
        alpha[int(key)] = value
    r2 = x @ x
    return (special.gammaln((d + nu) / 2.) - special.gammaln(nu / 2.)
            - d / 2. * np.log(nu * np.pi) - (d + nu) / 2. * np.log1p(r2 / nu)
            + np.log(2.) + special.log_ndtr((alpha @ x) * np.sqrt((d + nu) / (nu + r2))))


class NumbaAsymmetryTests(unittest.TestCase):
    def test_extended_density_and_coordinate_match_physical_target_and_jax(self):
        from dcs_bench import scs_paper
        scs_paper._author()
        import jax.numpy as jnp

        rng = np.random.default_rng(93822)
        for d in (6, 100):
            for beta in (1., 2.):
                spec = target_spec(d)
                mapping = BallMap(d, beta=beta, scale=1.4,
                    mu=np.linspace(-.3, .4, d), delta=np.r_[.2, -.1, np.zeros(d - 2)])
                args = native._inputs(mapping.parameters(), {'target': spec})
                for radius in (0., .7, .999999):
                    direction = rng.normal(size=d)
                    state = direction * radius / np.linalg.norm(direction)
                    x, jac = mapping.forward(state)
                    actual, norm, first, second, third = native._evaluate_extended(state, *args)
                    expected = direct_log_prob(spec, x) + jac
                    jax_expected = float(Target(spec).log_prob(jnp.asarray(x), 'jax')) + jac
                    with self.subTest(d=d, beta=beta, radius=radius):
                        self.assertAlmostEqual(actual, expected, delta=2e-7)
                        self.assertAlmostEqual(actual, jax_expected, delta=2e-7)
                        np.testing.assert_allclose([norm, first, second, third], [np.linalg.norm(x), *x[:3]],
                                                   rtol=5e-9, atol=1e-10)
        # Stable beta=2 cancellation also supports the closest representable
        # interior point; the naive map/Jacobian oracle loses this tiny gap.
        args = native._inputs(BallMap(100).parameters(), {'target': target_spec(100)})
        state = np.r_[np.nextafter(1., 0.), np.zeros(99)]
        self.assertTrue(np.all(np.isfinite(native._evaluate_extended(state, *args))))
        self.assertEqual(native._evaluate_extended(np.ones(100), *args)[0], -np.inf)

    def test_extra_measurements_preserve_legacy_seeded_trajectories(self):
        for d in (6, 100, 500):
            spec = target_spec(d)
            mapping = BallMap(d, scale=1.7,
                              delta=np.r_[.15, np.zeros(d - 1)])
            old_args = native._inputs(mapping.parameters(), {'target': spec})
            new_args = native._inputs(mapping.parameters(), {'target': spec})
            for reflected in (False, True):
                initial = mapping.inverse(np.ones(d))
                old_state, new_state = initial.copy(), initial.copy()
                old_lp, old_norm = native._evaluate(initial, *old_args)
                new_lp, new_norm, first, second, third = native._evaluate_extended(initial, *new_args)
                self.assertEqual(old_lp, new_lp)
                self.assertEqual(old_norm, new_norm)
                old = native._run_block(np.random.default_rng(724), old_state, old_lp, old_norm,
                    .4, 300, 10, 10., reflected, *old_args)
                new = native._run_block_extended(np.random.default_rng(724), new_state,
                    new_lp, new_norm, first, second, third, .4, 300, 10, 10., 3., reflected, *new_args)
                with self.subTest(d=d, reflected=reflected):
                    np.testing.assert_array_equal(old_state, new_state)
                    self.assertEqual(old[0], new[0])
                    self.assertEqual(old[1], new[1])
                    np.testing.assert_array_equal(old[2], new[5][:, :3])
                    np.testing.assert_array_equal(old[3], new[6][:, 0])

    def test_all_state_coordinate_summaries_match_independent_trajectory(self):
        spec = target_spec()
        mapping = BallMap(6, scale=1.6,
                          mu=np.linspace(-.1, .1, 6), delta=np.r_[.15, np.zeros(5)])
        args = native._inputs(mapping.parameters(), {'target': spec})
        for reflected in (False, True):
            state = mapping.inverse(np.ones(6))
            initial = state.copy()
            point, jac = mapping.forward(state)
            lp = direct_log_prob(spec, point) + jac
            rng = np.random.default_rng(531)
            values = []
            for _ in range(300):
                normals = rng.standard_normal(8)
                increment = .5 * normals[:6] / np.linalg.norm(normals)
                logu = np.log(rng.random())
                candidate = reflected_reference(state, increment) if reflected else state + increment
                if candidate @ candidate < 1.:
                    next_point, jac = mapping.forward(candidate)
                    next_lp = direct_log_prob(spec, next_point) + jac
                else:
                    next_lp = -np.inf
                accepted = logu < next_lp - lp
                if accepted:
                    state, point, lp = candidate, next_point, next_lp
                radius = np.linalg.norm(point)
                # Keep the threshold away from the deterministic initial X1=1:
                # independent map evaluations can round that value to opposite
                # sides of one, even when the underlying point is identical.
                values.append([radius >= 5., radius, accepted,
                               point[0], point[0] > 1.25,
                               point[1], point[1] < -1.25,
                               point[2], point[2] > 1.25])
            values = np.asarray(values)
            expected = np.cumsum(values, axis=0)[99::100]
            observed = []
            for thin in (1, 10):
                native_state = initial.copy()
                nlp, radius, first, second, third = native._evaluate_extended(initial, *args)
                result = native._run_block_extended(np.random.default_rng(531), native_state,
                    nlp, radius, first, second, third, .5, 300, thin, 5., 1.25, reflected, *args)
                np.testing.assert_allclose(result[5], expected, rtol=3e-10, atol=2e-9)
                np.testing.assert_array_equal(result[5][:, (0, 2, 4, 6, 8)], expected[:, (0, 2, 4, 6, 8)])
                # Repeated independent chord folds can amplify reduction-order
                # roundoff in individual positions. Decisions and tail counts
                # must agree exactly; seeded native parity is bitwise above.
                np.testing.assert_allclose(result[6], values[thin - 1::thin][:, (1, 3, 5, 7)],
                                           rtol=3e-10, atol=1e-7)
                np.testing.assert_allclose(native_state, state, rtol=3e-10, atol=1e-7)
                observed.append(result[5])
            np.testing.assert_array_equal(*observed)
            # Prefix subtraction removes exactly the first third of this batch;
            # no saved-trace thinning enters any coordinate estimate.
            np.testing.assert_allclose(observed[0][-1] - observed[0][0],
                                       values[100:].sum(axis=0), rtol=3e-10, atol=1e-7)

    def test_public_backend_caches_coordinate_and_preserves_calibration(self):
        spec = target_spec()
        mapping = BallMap(6, scale=1.7).parameters()
        state = BallMap(**mapping).inverse(np.ones(6))
        calibration = SimpleNamespace(state=state, scale=.4, compile_seconds=3.,
            warmup_seconds=2., warmup_acceptance=.8, tuning_metadata={})
        config = dict(target=spec, algorithm='DCS-BW-reflect', thin=10,
                      threshold=5., coordinate_metrics=True, coordinate_threshold=1.)
        first = native.NumbaRadial(mapping, config, 321, calibration)
        second = native.NumbaRadial(mapping, config, 321, calibration)
        for _ in range(2):
            a, b = first.batch(), second.batch()
            self.assertEqual(a[0].shape, (3, 9))
            self.assertEqual(a[1].shape, (30, 4))
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
            self.assertEqual(first.scale, calibration.scale)
        np.testing.assert_array_equal(calibration.state, state)
        self.assertEqual(calibration.tuning_metadata, {})

    def test_rejections_retain_all_coordinates_and_tails_are_strict(self):
        spec = target_spec()
        mapping = BallMap(6, mu=np.array([1., -1., 1., 0., 0., 0.]))
        args = native._inputs(mapping.parameters(), {'target': spec})
        state = np.zeros(6)
        lp, radius, first, second, third = native._evaluate_extended(state, *args)
        result = native._run_block_extended(np.random.default_rng(88), state,
            lp, radius, first, second, third, 1e20, 300, 10, 5., 1., False, *args)
        np.testing.assert_array_equal(state, np.zeros(6))
        # Every proposal leaves the ball, so all 300 repeated states contribute.
        np.testing.assert_array_equal(result[5][:, (0, 2, 4, 6, 8)], np.zeros((3, 5)))
        np.testing.assert_array_equal(result[5][:, (3, 5, 7)],
                                      np.array([[100., -100., 100.],
                                                [200., -200., 200.],
                                                [300., -300., 300.]]))
        np.testing.assert_array_equal(result[6][:, 1:], np.tile([1., -1., 1.], (30, 1)))


if __name__ == '__main__':
    unittest.main()
