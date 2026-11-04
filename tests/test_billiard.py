"""Independent billiard geometry, volume, calibration and recording checks."""
import copy
import unittest

import numpy as np

from dcs_bench.billiard import reflect_ball_billiard
from dcs_bench.ball_boundary import reflect_ball_proposal
from dcs_bench import timed_radial as timed
from dcs_bench.transforms import BallMap
from dcs_bench.targets import Target


def bounce_reference(start, increment):
    """Slow collision-by-collision oracle; deliberately unlike the closed form."""
    point = np.asarray(start, dtype=float).copy()
    length = np.linalg.norm(increment)
    if length == 0:
        return point, np.asarray(increment)
    velocity = np.asarray(increment) / length
    impact2 = np.dot(point, point) - np.dot(point, velocity)**2
    half = np.sqrt(1. - impact2)
    angle = 2. * np.arcsin(half)
    extra = np.floor(max(0., np.arctan(length / (2. * half)) / angle - length / (2. * half)))
    remaining = length + 2. * half * extra
    for _ in range(10000):
        dot = np.dot(point, velocity)
        hit_distance = -dot + np.sqrt(max(0., dot*dot + 1. - np.dot(point, point)))
        if remaining < hit_distance:
            return point + remaining * velocity, length * velocity
        point += hit_distance * velocity
        point /= np.linalg.norm(point)
        velocity -= 2. * np.dot(velocity, point) * point
        remaining -= hit_distance
    raise AssertionError('Oracle exceeded its collision budget')


class BilliardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        jax.config.update('jax_enable_x64', True)

    def test_matches_independent_bounces_reverses_and_preserves_proposed_radius(self):
        import jax

        rng = np.random.default_rng(713219)
        for dimension in (2, 5, 100):
            count = 64
            draw = rng.normal(size=(count, dimension+2))
            points = draw[:, :dimension] / np.linalg.norm(draw, axis=1)[:, None]
            draw = rng.normal(size=(count, dimension+2))
            increments = draw[:, :dimension] / np.linalg.norm(draw, axis=1)[:, None]
            increments *= np.resize([.03, .3, .8, 2.4], count)[:, None]
            actual, valid = jax.jit(jax.vmap(reflect_ball_billiard))(points, increments)
            folded, folded_valid = jax.jit(jax.vmap(reflect_ball_proposal))(points, increments)
            self.assertTrue(np.all(valid))
            self.assertTrue(np.all(folded_valid))
            np.testing.assert_allclose(np.sum(np.asarray(actual)**2, axis=1),
                np.sum(np.asarray(folded)**2, axis=1), atol=2e-13, rtol=2e-13)
            reference = [bounce_reference(x, v) for x, v in zip(points, increments)]
            expected, velocities = np.array([r[0] for r in reference]), np.array([r[1] for r in reference])
            np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
            reversed_points, reversed_valid = jax.jit(jax.vmap(reflect_ball_billiard))(expected, -velocities)
            self.assertTrue(np.all(reversed_valid))
            np.testing.assert_allclose(reversed_points, points, atol=2e-10, rtol=2e-10)

    def test_extends_inside_proposals_and_handles_diameter_grazing_and_invalid_inputs(self):
        import jax

        # An interior raw endpoint may still need invariant extra chords.
        # Applying the extension only at raw boundary crossings is not reversible.
        point = np.array([.999, 0.])
        step = np.array([0., .02])
        self.assertLess(np.linalg.norm(point+step), 1.)
        end, valid = reflect_ball_billiard(point, step)
        self.assertTrue(valid)
        self.assertGreater(np.linalg.norm(np.asarray(end)-(point+step)), .01)
        np.testing.assert_allclose(end, bounce_reference(point, step)[0], atol=1e-12)
        for x, increment in [(np.zeros(2), np.zeros(2)),
                              (np.array([.2, 0.]), np.array([7.4, 0.])),
                              (np.array([1.-1e-12, 0.]), np.array([0., np.exp(15.)]))]:
            end, valid = reflect_ball_billiard(x, increment)
            self.assertTrue(valid)
            self.assertTrue(np.all(np.isfinite(end)))
            self.assertLess(np.linalg.norm(end), 1.)
        for increment in ([np.inf, 0.], [np.nan, .1], [1., 0.], [1e300, 0.]):
            x = np.zeros(2)
            end, valid = reflect_ball_billiard(x, np.array(increment))
            self.assertFalse(valid)
            np.testing.assert_array_equal(end, x)
        # The full supported calibration range remains representable in500D.
        rng = np.random.default_rng(51623)
        draw = rng.normal(size=(128, 502))
        starts = draw[:, :500] / np.linalg.norm(draw, axis=1)[:, None]
        draw = rng.normal(size=(128, 502))
        increments = draw[:, :500] / np.linalg.norm(draw, axis=1)[:, None]
        increments *= np.exp(np.linspace(-30., 15., 128))[:, None]
        ends, valid = jax.jit(jax.vmap(reflect_ball_billiard))(starts, increments)
        self.assertTrue(np.all(valid))
        self.assertTrue(np.all(np.linalg.norm(ends, axis=1) < 1.))

    def test_uniform_volume_and_joint_transition_symmetry(self):
        import jax

        rng = np.random.default_rng(37923)
        count = 16000
        for dimension in (2, 5, 100):
            draw = rng.normal(size=(count, dimension+2))
            starts = draw[:, :dimension] / np.linalg.norm(draw, axis=1)[:, None]
            draw = rng.normal(size=(count, dimension+2))
            increments = .4*draw[:, :dimension] / np.linalg.norm(draw, axis=1)[:, None]
            ends, valid = jax.jit(jax.vmap(reflect_ball_billiard))(starts, increments)
            self.assertTrue(np.all(valid))
            ends = np.asarray(ends)
            radial_cdf = np.linalg.norm(ends, axis=1)**dimension
            self.assertLess(abs(radial_cdf.mean()-.5), 6./np.sqrt(12.*count))
            second = 1./(dimension+2)
            np.testing.assert_array_less(np.abs(ends.mean(axis=0)), 6.*np.sqrt(second/count))
            variance_square = 3./((dimension+2)*(dimension+4))-second**2
            np.testing.assert_array_less(np.abs(np.mean(ends**2, axis=0)-second),
                6.*np.sqrt(variance_square/count))
            # Equal-width bins on a scaled coordinate, with under/overflow bins.
            cuts = np.array([-np.inf, -1., -.3, .3, 1., np.inf])
            left = np.digitize(starts[:, 0]*np.sqrt(dimension+2), cuts)-1
            right = np.digitize(ends[:, 0]*np.sqrt(dimension+2), cuts)-1
            histogram = np.zeros((5, 5), dtype=int)
            np.add.at(histogram, (left, right), 1)
            for i in range(5):
                for j in range(i+1, 5):
                    self.assertLessEqual(abs(histogram[i, j]-histogram[j, i]),
                        6.*np.sqrt(histogram[i, j]+histogram[j, i])+1.)


class TimedBilliardTests(unittest.TestCase):
    def config(self):
        return dict(algorithm=timed.BILLIARD_METHOD, dimension=5, nu=3., alpha={0: 3., 1: -3.},
            thin=10, warmup=18, threshold=50., coordinate_metrics=True, coordinate_threshold=3.,
            time_budget=.001, dcs_backend='jax')

    def test_effective_map_preserves_joint_fit_and_exact_reflected_pilot(self):
        mapping = BallMap(5, mu=[.3, -.2, 0., 0., 0.], scale=1.7, delta=[.2, -.1, 0., 0., 0.]).parameters()
        original = copy.deepcopy(mapping)
        effective = timed.effective_mapping(mapping, timed.BILLIARD_METHOD)
        self.assertEqual(mapping, original)
        self.assertEqual(effective['delta'], [0.]*5)
        for field in ('mu', 'scale', 'beta', 'dimension'):
            self.assertEqual(effective[field], mapping[field])
        config = self.config()
        spec = timed.scs_paper.target_spec(config)
        _, projection, initial, density = timed._components(mapping, spec, timed.BILLIARD_METHOD)
        np.testing.assert_allclose(projection(initial), np.ones(5), atol=1e-12)
        reference_map = BallMap(**effective)
        target = Target(spec)
        for point in (np.zeros(5), np.array([.5, -.2, .1, 0., 0.]),
                      np.array([-.7, .3, -.2, .1, .1])):
            physical, jacobian = reference_map.forward(point)
            np.testing.assert_allclose(projection(point), physical, atol=1e-12)
            np.testing.assert_allclose(density(point), target.log_prob(physical)+jacobian, atol=1e-11)
            self.assertFalse(np.allclose(projection(point), BallMap(**mapping).forward(point)[0]))
        candidate = timed._backend(mapping, config, 7134)
        control = timed._backend(effective, {**config, 'algorithm': 'DCS-BW-reflect',
            'step_tuning_policies': {'DCS-BW-reflect': timed.BALL_TUNING}}, 7134)
        self.assertEqual(candidate.scale, control.scale)
        self.assertEqual(candidate.tuning_metadata['tuning_history'], control.tuning_metadata['tuning_history'])
        np.testing.assert_array_equal(candidate.state, control.state)
        self.assertEqual(candidate.tuning_metadata['effective_map'], effective)
        self.assertEqual(candidate.tuning_metadata['tuning_pilot_iterations'], config['warmup'])
        prefix, saved = candidate.batch()
        self.assertEqual(prefix.shape, (3, 9))
        self.assertEqual(saved.shape, (30, 4))
        self.assertTrue(np.all(np.isfinite(prefix)))
        self.assertTrue(np.all(np.isfinite(saved)))
        self.assertEqual(mapping, original)

    def test_vi_off_direct_components_and_timed_metadata(self):
        import jax.numpy as jnp
        config = self.config()
        mapping = BallMap(5).parameters()
        self.assertEqual(timed.effective_mapping(mapping, timed.BILLIARD_METHOD), mapping)
        _, project, initial, density = timed._components(mapping,
            timed.scs_paper.target_spec(config), timed.BILLIARD_METHOD)
        np.testing.assert_allclose(project(initial), np.ones(5), atol=1e-12)
        self.assertTrue(np.isfinite(float(density(jnp.asarray(initial)))))
        traces, record = timed.run_timed(mapping, config, 771)
        self.assertEqual(record['backend'], 'JAX')
        self.assertEqual(record['boundary_policy'], 'specular_ball_billiard_invariant_extra_chords')
        self.assertEqual(record['calibration_algorithm'], 'DCS-BW-reflect')
        self.assertEqual(record['effective_map'], mapping)
        self.assertFalse(record['bounce_loop'])
        self.assertFalse(record['adaptation_during_sampling'])
        self.assertEqual(record['discarded_iterations']*3, record['iterations'])
        self.assertEqual(len(record['mean_all_iterations']), 8)
        self.assertTrue(np.all(np.isfinite(traces)))

    def test_rejects_native_backend_and_alternative_pilot(self):
        config = self.config()
        native = {**config, 'dcs_backend': 'numba'}
        with self.assertRaisesRegex(ValueError, 'JAX'):
            timed._validate(native)
        with self.assertRaisesRegex(ValueError, 'JAX'):
            timed._backend(BallMap(5).parameters(), native, 1)
        with self.assertRaisesRegex(ValueError, 'pilot'):
            timed.resolve_step_tuning({**config,
                'step_tuning_policies': {timed.BILLIARD_METHOD: timed.ACCEPTANCE_TUNING}})


if __name__ == '__main__':
    unittest.main()
