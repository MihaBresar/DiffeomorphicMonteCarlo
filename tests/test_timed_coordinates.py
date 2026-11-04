"""Coordinate estimators must use the same complete post-MH history as radii."""
import unittest
from unittest.mock import patch

import numpy as np

from dcs_bench import timed_radial as timed
from dcs_bench.transforms import BallMap, SphereMap


class TimedCoordinateTests(unittest.TestCase):
    def test_coordinate_record_uses_every_retained_iteration_with_exact_third_burn(self):
        radii = np.repeat(np.linspace(1., 80., 300), 2)
        x1 = np.repeat(np.linspace(-5., 5., 300), 2)
        accepted = np.tile([1., 0.], 300)
        values = np.column_stack((radii >= 50., radii, accepted, x1, x1 > 3.))

        class Fake:
            compile_seconds, warmup_seconds, warmup_acceptance = 4., 3., .2
            scale, implementation, index = .125, 'test', 0

            def batch(self):
                selected = slice(self.index*300, (self.index+1)*300)
                self.index += 1
                prefixes = values[selected].reshape(3, 100, 5).sum(axis=1).cumsum(axis=0)
                saved = np.column_stack((radii[selected], x1[selected]))[thin-1::thin]
                return prefixes, saved

        for thin in (1, 10, 50):
            config = dict(algorithm='DCS-BW', time_budget=.015, warmup=100, thin=thin,
                          coordinate_metrics=True, coordinate_threshold=3.)
            with self.subTest(thin=thin), patch.object(timed, '_backend', return_value=Fake()), \
                    patch.object(timed.time, 'perf_counter', side_effect=[0., .01, .02]), \
                    patch.object(timed.time, 'process_time', side_effect=[8., 8.01]):
                traces, record = timed.run_timed({}, config, 8)
            expected = values[200:][:, [0, 1, 3, 4]]
            np.testing.assert_allclose(record['mean_all_iterations'], expected.mean(axis=0))
            np.testing.assert_allclose(traces, expected[thin-1::thin])
            self.assertEqual(record['x1_tail_count'], int((x1[200:] > 3.).sum()))
            self.assertAlmostEqual(record['x1_sum'], x1[200:].sum())
            self.assertEqual(record['postburn_accepted'], 200)
            self.assertEqual(record['discarded_iterations'], 200)
            self.assertEqual(record['estimation_iterations'], 400)
            self.assertEqual(record['functionals'],
                             ['tail_probability', 'mean_norm', 'mean_x1', 'x1_tail_probability'])

    def test_coordinate_recording_preserves_jax_transition_and_calibration(self):
        import jax
        for algorithm in ('DCS-RWM', 'DCS-BW', 'DCS-BW-reflect', 'SCS'):
            mapping = (SphereMap(5, scale=1.7) if algorithm == 'SCS' else BallMap(5, scale=1.7)).parameters()
            config = dict(algorithm=algorithm, dimension=5, nu=3., alpha={0: 3., 1: -3.},
                          thin=10, warmup=60,
                          step_tuning_policies={'DCS-BW-reflect': timed.BALL_TUNING})
            old = timed._Matched(mapping, config, 774)
            extended = timed._Matched(mapping, {**config, 'coordinate_metrics': True,
                                                'coordinate_threshold': 3.}, 774)
            with self.subTest(algorithm=algorithm):
                self.assertEqual(old.scale, extended.scale)
                np.testing.assert_array_equal(old.state, extended.state)
                block, projection, _, _ = timed._components(mapping, timed.scs_paper.target_spec(config), algorithm)
                _, key = jax.random.split(extended.key)
                states, accepted = block(extended.state, key, 300, extended.scale)
                physical = np.asarray(projection(states))
                prefixes, saved = extended.batch()
                legacy, norms = old.batch()
                np.testing.assert_allclose(extended.state, old.state, rtol=1e-13, atol=1e-14)
                np.testing.assert_allclose(prefixes[:, :3], legacy, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(saved[:, 0], norms, rtol=1e-12)
                extra = np.column_stack((physical[:, 0], physical[:, 0] > 3.))
                expected = extra.reshape(3, 100, 2).sum(axis=1).cumsum(axis=0)
                np.testing.assert_allclose(prefixes[:, 3:], expected, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(saved[:, 1], physical[9::10, 0], rtol=1e-12)

    def test_coordinate_configuration_errors_fail_before_sampling(self):
        base = dict(algorithm='SCS', time_budget=.05, warmup=100, thin=10, coordinate_metrics=True)
        for bad in (None, float('nan'), float('inf'), True):
            with self.subTest(threshold=bad), patch.object(timed, '_backend') as backend:
                with self.assertRaises(ValueError):
                    timed.run_timed({}, {**base, 'coordinate_threshold': bad}, 1)
                backend.assert_not_called()


if __name__ == '__main__':
    unittest.main()
