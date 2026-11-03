"""Geometry and invariant-volume checks for reflected-chord ball proposals."""
import unittest

import numpy as np

from dcs_bench.ball_boundary import reflect_ball_proposal, reflect_ball_step


class BallBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        jax.config.update('jax_enable_x64', True)

    def test_inside_proposals_are_unchanged_and_diameter_reflections_are_exact(self):
        import jax.numpy as jnp

        x = jnp.asarray([.2, -.1, .3])
        increment = jnp.asarray([.1, .15, -.2])
        np.testing.assert_array_equal(reflect_ball_step(x, increment), x + increment)
        np.testing.assert_array_equal(reflect_ball_step(x, jnp.zeros(3)), x)
        for start, displacement, expected in ((.2, 1.1, .7), (.2, -1.5, -.7),
                                               (.2, 4.1, .3), (.2, 7.4, -.4),
                                               (-.3, -8.2, -.5)):
            with self.subTest(start=start, displacement=displacement):
                result = reflect_ball_step(jnp.asarray([start, 0.]),
                                           jnp.asarray([displacement, 0.]))
                np.testing.assert_allclose(result, [expected, 0.], atol=2e-15, rtol=0)

    def test_noncentral_chord_keeps_transverse_coordinate_and_handles_many_bounces(self):
        import jax.numpy as jnp

        # At transverse coordinate .6, the chord endpoints are -.8 and .8.
        x = jnp.asarray([.1, .6, 0.])
        for displacement, expected in ((1.0, .5), (-1.2, -.5),
                                         (3.4, .3), (32.4, .5)):
            with self.subTest(displacement=displacement):
                result = reflect_ball_step(x, jnp.asarray([displacement, 0., 0.]))
                np.testing.assert_allclose(result, [expected, .6, 0.], atol=1e-14, rtol=0)
                self.assertLess(np.linalg.norm(result), 1.)

    def test_invalid_or_exact_boundary_results_self_loop_without_projection(self):
        import jax.numpy as jnp

        x = jnp.asarray([0., 0.])
        for increment in ([1., 0.], [-1., 0.], [3., 0.],
                          [np.inf, 0.], [np.nan, .1]):
            with self.subTest(increment=increment):
                np.testing.assert_array_equal(reflect_ball_step(x, jnp.asarray(increment)), x)
                point, valid = reflect_ball_proposal(x, jnp.asarray(increment))
                np.testing.assert_array_equal(point, x)
                self.assertFalse(bool(valid))
        _, zero_valid = reflect_ball_proposal(x, jnp.zeros(2))
        self.assertTrue(bool(zero_valid))

    def test_adaptation_ceiling_and_grazing_chords_remain_finite(self):
        import jax

        rng = np.random.default_rng(171901)
        count, dimension = 4096, 100
        gaussian = rng.normal(size=(count, dimension + 2))
        points = gaussian[:, :dimension] / np.linalg.norm(gaussian, axis=1)[:, None]
        increments = rng.normal(size=(count, dimension + 2))
        increments = (np.exp(15.) * increments[:, :dimension]
                      / np.linalg.norm(increments, axis=1)[:, None])
        endpoint, valid = jax.jit(jax.vmap(reflect_ball_proposal))(points, increments)
        self.assertTrue(np.all(np.asarray(valid)))
        self.assertTrue(np.all(np.isfinite(endpoint)))
        self.assertTrue(np.all(np.linalg.norm(endpoint, axis=1) < 1.))
        # A tangential move close to the boundary has a short positive chord.
        point, valid = reflect_ball_proposal(np.array([1. - 1e-12, 0.]),
                                             np.array([0., np.exp(15.)]))
        self.assertTrue(np.all(np.isfinite(point)))
        self.assertLess(np.linalg.norm(point), 1.)

    def test_uniform_ball_volume_is_preserved_in_low_and_high_dimensions(self):
        import jax

        count = 25000
        rng = np.random.default_rng(714016)

        def uniform_ball(dimension):
            gaussian = rng.normal(size=(count, dimension + 2))
            return gaussian[:, :dimension] / np.linalg.norm(gaussian, axis=1)[:, None]

        for dimension in (1, 5, 100):
            with self.subTest(dimension=dimension):
                points = uniform_ball(dimension)
                step = jax.jit(jax.vmap(reflect_ball_proposal))
                crossings = 0
                # Large displacements exercise multiple reflections, not only
                # the ordinary inside-ball fast path.
                for scale in (.5, 1.7, 8.):
                    increments = scale * uniform_ball(dimension)
                    crossings += np.count_nonzero(np.linalg.norm(points + increments, axis=1) >= 1.)
                    proposal, valid = step(points, increments)
                    self.assertTrue(np.all(np.asarray(valid)))
                    points = np.asarray(proposal)
                    self.assertTrue(np.all(np.isfinite(points)))
                    radii = np.linalg.norm(points, axis=1)
                    self.assertTrue(np.all(radii < 1.))
                    radial_cdf = radii ** dimension
                    self.assertLess(abs(radial_cdf.mean() - .5), 6 / np.sqrt(12 * count))
                    for probability in (.1, .25, .5, .75, .9):
                        self.assertLess(abs(np.mean(radial_cdf <= probability) - probability),
                            6 * np.sqrt(probability * (1 - probability) / count) + 1 / count)
                    second = 1 / (dimension + 2)
                    np.testing.assert_array_less(np.abs(points.mean(axis=0)),
                                                 6 * np.sqrt(second / count))
                    variance_square = 3 / ((dimension + 2) * (dimension + 4)) - second ** 2
                    np.testing.assert_array_less(np.abs(np.mean(points ** 2, axis=0) - second),
                                                 6 * np.sqrt(variance_square / count))
                self.assertGreater(crossings, count)

    def test_conditional_interval_joint_transition_counts_are_symmetric(self):
        import jax

        # Along a fixed unoriented chord the signed step has the even radial
        # density d*|s|^(d-1)/(2*h^d), rather than a uniform signed step for d>1.
        # Start uniformly on the chord and check P(X in A,Y in B)=P(X in B,Y in A).
        rng = np.random.default_rng(225017)
        count, dimension, scale, bins = 180000, 5, 2.3, 8
        starts = rng.uniform(-1., 1., count)
        signs = rng.choice([-1., 1.], count)
        steps = signs * scale * rng.random(count) ** (1 / dimension)
        ends = np.asarray(jax.jit(jax.vmap(reflect_ball_step))(
            starts[:, None], steps[:, None]))[:, 0]
        histogram, _, _ = np.histogram2d(starts, ends, bins=bins,
                                        range=[[-1., 1.], [-1., 1.]])
        for i in range(bins):
            for j in range(i + 1, bins):
                combined = histogram[i, j] + histogram[j, i]
                # Conditional on pair membership, its orientation is Bernoulli
                # .5 under reversibility; difference variance is combined.
                self.assertLessEqual(abs(histogram[i, j] - histogram[j, i]),
                                     6 * np.sqrt(combined) + 1)


if __name__ == '__main__':
    unittest.main()
