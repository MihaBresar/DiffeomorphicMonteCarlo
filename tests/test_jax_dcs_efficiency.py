"""Distributional and numerical checks for transition-preserving JAX shortcuts."""
import unittest

import numpy as np
from scipy.special import gammaln, log_ndtr

from dcs_bench import scs_paper, timed_radial as timed
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap


def beta_two_reference(mapping, target, point):
    """Independent extended-precision evaluation, including the boundary gap.

    The exact ball automorphism identity avoids subtracting a rounded mapped
    norm from one. NumPy longdouble may equal double on some platforms; the
    extreme-boundary assertions therefore allow ordinary float64 roundoff.
    """
    z, delta, mu = (np.asarray(value, dtype=np.longdouble)
                    for value in (point, mapping.delta, mapping.mu))
    scale, nu = np.longdouble(mapping.scale), np.longdouble(target.nu)
    r2, delta2, dot = z @ z, delta @ delta, z @ delta
    denominator = 1 + 2 * dot + delta2 * r2
    w = ((1 - delta2) * z + (1 + 2 * dot + r2) * delta) / denominator
    gap = (1 - delta2) * (1 - r2) / denominator
    e = np.sqrt(gap)
    physical = mu + scale * w / e
    # Cancel the radial denominator algebraically before evaluating t density.
    scaled_x = mu * e + scale * w
    s = nu * gap + scaled_x @ scaled_x
    d = target.d
    constant = (np.log(2.) + gammaln((d + target.nu) / 2)
                - gammaln(target.nu / 2) + nu / 2 * np.log(nu)
                - d / 2 * np.log(np.longdouble(np.pi)))
    logjac_mobius = d * (np.log1p(-delta2) - np.log(denominator))
    argument = np.asarray(target.alpha, dtype=np.longdouble) @ scaled_x
    argument *= np.sqrt((nu + d) / s)
    density = (constant + (nu / 2 - 1) * np.log(gap) + d * np.log(scale)
               - (nu + d) / 2 * np.log(s) + logjac_mobius
               + log_ndtr(float(argument)))
    return np.asarray(physical, dtype=float), float(density)


class JAXDCSEfficiencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scs_paper._author()  # Common float64 policy, also used by production.

    def test_uniform_ball_has_correct_radius_and_isotropic_coordinates(self):
        import jax

        count = 20000
        for dimension in (1, 5, 100):
            with self.subTest(dimension=dimension):
                keys = jax.random.split(jax.random.key(8400 + dimension), count)
                draws = np.asarray(jax.jit(jax.vmap(
                    lambda key: timed._uniform_ball(key, dimension)))(keys))
                self.assertEqual(draws.shape, (count, dimension))
                self.assertTrue(np.all(np.isfinite(draws)))
                radii = np.linalg.norm(draws, axis=1)
                self.assertTrue(np.all((radii > 0) & (radii < 1)))

                # A uniform d-ball has R^d ~ Uniform(0,1). This detects shell
                # proposals and incorrect powers even when coordinates look
                # isotropic. Six-standard-error bounds avoid brittle RNG tests.
                cdf_values = radii ** dimension
                self.assertLess(abs(cdf_values.mean() - .5), 6 / np.sqrt(12 * count))
                self.assertLess(abs(np.mean(cdf_values ** 2) - 1 / 3),
                                6 * np.sqrt(4 / (45 * count)))
                for probability in (.1, .25, .5, .75, .9):
                    observed = np.mean(cdf_values <= probability)
                    tolerance = 6 * np.sqrt(probability * (1 - probability) / count) + 1 / count
                    self.assertLess(abs(observed - probability), tolerance)

                second_moment = 1 / (dimension + 2)
                np.testing.assert_array_less(np.abs(draws.mean(axis=0)),
                                             6 * np.sqrt(second_moment / count))
                variance_of_square = 3 / ((dimension + 2) * (dimension + 4)) - second_moment ** 2
                np.testing.assert_array_less(
                    np.abs(np.mean(draws ** 2, axis=0) - second_moment),
                    6 * np.sqrt(variance_of_square / count))
                np.testing.assert_array_less(np.abs(np.mean(draws > 0, axis=0) - .5),
                                             6 * np.sqrt(.25 / count))
                if dimension > 1:
                    cross = np.mean(draws[:, 0] * draws[:, -1])
                    self.assertLess(abs(cross),
                        6 * np.sqrt(1 / ((dimension + 2) * (dimension + 4) * count)))

    def test_beta_two_projection_and_density_match_canonical_fitted_maps(self):
        import jax.numpy as jnp

        for dimension in (3, 100):
            target = Target(dict(kind='skew_t', dimension=dimension, nu=3.,
                                 alpha={0: 100., 1: -100.}))
            shift = np.zeros(dimension)
            shift[:3] = [.19, -.13, .07]
            mapping = BallMap(dimension, beta=2., mu=np.linspace(-.3, .4, dimension),
                              scale=1.37, delta=shift)
            _, projection, _, density = timed._components(mapping.parameters(), target.spec, 'DCS-BW')
            points = []
            for radius in (0., .5, .99, .999999):
                point = np.zeros(dimension)
                point[0] = radius
                points.append(point)
                with self.subTest(dimension=dimension, radius=radius):
                    expected_x, _ = mapping.forward(point)
                    expected_lp = mapping.log_density(target, point)
                    np.testing.assert_allclose(projection(jnp.asarray(point)), expected_x,
                                               rtol=3e-9, atol=2e-10)
                    self.assertAlmostEqual(float(density(jnp.asarray(point))), expected_lp, delta=1e-6)
            # The reporting path projects a matrix, whereas the sampler uses
            # vectors. Both must implement the same transform.
            batch_x = np.asarray(projection(jnp.asarray(points)))
            for row, point in zip(batch_x, points):
                np.testing.assert_allclose(row, projection(jnp.asarray(point)), rtol=3e-12, atol=1e-12)

    def test_beta_two_gap_stays_finite_near_boundary_and_large_mobius_shift(self):
        import jax.numpy as jnp

        dimension = 100
        target = Target(dict(kind='skew_t', dimension=dimension, nu=3.,
                             alpha={0: 100., 1: -100.}))
        for shift_size in (.3, .999):
            shift = np.zeros(dimension)
            shift[0] = shift_size
            mapping = BallMap(dimension, beta=2., mu=np.linspace(-.2, .2, dimension),
                              scale=1.41, delta=shift)
            _, projection, _, density = timed._components(mapping.parameters(), target.spec, 'DCS-BW')
            for distance in (1e-6, 1e-10, 1e-12):
                point = np.zeros(dimension)
                point[0] = -(1 - distance)
                with self.subTest(shift=shift_size, distance=distance):
                    expected_x, expected_lp = beta_two_reference(mapping, target, point)
                    actual_x = np.asarray(projection(jnp.asarray(point)))
                    actual_lp = float(density(jnp.asarray(point)))
                    self.assertTrue(np.all(np.isfinite(actual_x)))
                    self.assertTrue(np.isfinite(actual_lp))
                    # At 1-r ~ 1e-12, float64 subtraction can lose four to five
                    # relative digits even with an exact algebraic identity.
                    np.testing.assert_allclose(actual_x, expected_x, rtol=2e-4, atol=1e-8)
                    self.assertAlmostEqual(actual_lp, expected_lp, delta=.02)

    def test_nonquadratic_radial_exponents_keep_canonical_transform(self):
        import jax.numpy as jnp

        dimension = 7
        target = Target(dict(kind='skew_t', dimension=dimension, nu=3., alpha={0: 3., 1: -2.}))
        for beta in (.2, 1., 3.7):
            mapping = BallMap(dimension, beta=beta, mu=np.linspace(-.1, .2, dimension),
                              scale=1.7, delta=np.array([.1, -.05, 0., 0., 0., 0., 0.]))
            _, projection, _, density = timed._components(mapping.parameters(), target.spec, 'DCS-RWM')
            for radius in (0., .5, .95):
                point = np.zeros(dimension)
                point[0] = radius
                with self.subTest(beta=beta, radius=radius):
                    expected_x, _ = mapping.forward(point)
                    np.testing.assert_allclose(projection(jnp.asarray(point)), expected_x,
                                               rtol=3e-11, atol=1e-12)
                    self.assertAlmostEqual(float(density(jnp.asarray(point))),
                                           mapping.log_density(target, point), delta=1e-8)

    def test_outside_proposals_repeat_current_state_without_nonfinite_values(self):
        import jax
        import jax.numpy as jnp

        target = Target(dict(kind='skew_t', dimension=5, nu=3., alpha={0: 3.}))
        mapping = BallMap(5, beta=2., scale=1.7)
        initial = jnp.zeros(5)
        for algorithm in ('DCS-RWM', 'DCS-BW'):
            with self.subTest(algorithm=algorithm):
                block, _, _, density = timed._components(mapping.parameters(), target.spec, algorithm)
                states, accepted = block(initial, jax.random.key(784), 100, 1e6)
                np.testing.assert_array_equal(states, np.zeros((100, 5)))
                np.testing.assert_array_equal(accepted, np.zeros(100))
                self.assertEqual(float(density(jnp.ones(5))), -np.inf)


if __name__ == '__main__':
    unittest.main()
