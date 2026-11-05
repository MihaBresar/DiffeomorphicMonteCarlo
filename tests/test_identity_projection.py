"""Independent radial-map references guard the exact-zero JAX fast path."""
import math
import unittest

import numpy as np

from dcs_bench import timed_radial as timed
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap


class IdentityProjectionTests(unittest.TestCase):
    def test_zero_deformation_matches_independent_numpy_map_and_target(self):
        timed.scs_paper._author()
        import jax
        import jax.numpy as jnp
        for dimension in (100, 500):
            target = Target(dict(kind='skew_t', dimension=dimension, nu=3., alpha={'0': 10.}))
            states = np.zeros((5, dimension))
            states[:, 0] = [0., 1e-100, .3, .9, 1-1e-7]
            for beta in (.2, 1., 2.):
                mapping = BallMap(dimension, beta=beta, scale=1.7,
                                  mu=np.linspace(-.03, .03, dimension))
                _, project, _, density = timed._components(mapping.parameters(), target.spec, 'DCS-BW')
                expected, logjac = mapping.forward(states)
                actual = np.asarray(jax.jit(project)(jnp.asarray(states)))
                np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-12)
                expected_logdensity = target.log_prob(expected)+logjac
                observed = np.asarray(jax.jit(jax.vmap(density))(jnp.asarray(states)))
                self.assertTrue(np.isfinite(observed).all())
                np.testing.assert_allclose(observed, expected_logdensity, rtol=2e-11, atol=2e-8)

    def test_uniform_ball_student_t_density_is_constant_up_to_the_boundary(self):
        timed.scs_paper._author()
        import jax
        import jax.numpy as jnp
        for dimension in (100, 500):
            target = dict(kind='skew_t', dimension=dimension, nu=2., alpha={})
            mapping = BallMap(dimension, beta=2., scale=np.sqrt(2.)).parameters()
            _, _, _, density = timed._components(mapping, target, 'DCS-BW')
            states = np.zeros((7, dimension))
            states[:, 0] = [0., .2, .7, 1-1e-6, 1-1e-12, np.nextafter(1., 0.), 1.]
            observed = np.asarray(jax.jit(jax.vmap(density))(jnp.asarray(states)))
            expected = math.lgamma(dimension/2+1)-dimension/2*math.log(math.pi)
            np.testing.assert_allclose(observed[:-1], expected, rtol=2e-11, atol=2e-9)
            self.assertEqual(observed[-1], -np.inf)

    def test_nonzero_deformation_retains_independent_map_and_density(self):
        timed.scs_paper._author()
        import jax
        import jax.numpy as jnp
        dimension = 7
        target = Target(dict(kind='skew_t', dimension=dimension, nu=3., alpha={'0': 5.}))
        states = np.zeros((4, dimension))
        states[:, 0] = [0., .4, .9, 1-1e-6]
        for delta in (1e-150, .12):
            mapping = BallMap(dimension, scale=1.7, delta=[delta]+[0.]*(dimension-1))
            _, project, _, density = timed._components(mapping.parameters(), target.spec, 'DCS-RWM')
            expected, logjac = mapping.forward(states)
            np.testing.assert_allclose(jax.jit(project)(jnp.asarray(states)), expected, rtol=2e-9, atol=1e-12)
            observed = np.asarray(jax.jit(jax.vmap(density))(jnp.asarray(states)))
            np.testing.assert_allclose(observed, target.log_prob(expected)+logjac, rtol=2e-10, atol=2e-8)


if __name__ == '__main__':
    unittest.main()
