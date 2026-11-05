"""Supplied fixed SCS scales use the original author transition unchanged."""
import unittest
from unittest.mock import patch

import numpy as np

from dcs_bench import timed_radial as timed
from dcs_bench.transforms import SphereMap


def configuration(dimension=100, step=.05):
    return dict(algorithm='SCS',
        target=dict(kind='skew_t', dimension=dimension, nu=3., alpha={'0': 5.}),
        warmup=16, thin=1, time_budget=.001, threshold=50., initial_scale=.1,
        ess_enabled=False, coordinate_metrics=False, scs_fixed_move=step,
        step_tuning_policies={'SCS': timed.FIXED_SCS_TUNING})


class FixedSCSStepTests(unittest.TestCase):
    def test_invalid_method_scale_and_discarded_budget_are_rejected(self):
        for step in (None, 0., -.1, np.inf, np.nan, True, '.05'):
            with self.subTest(step=step), self.assertRaises(ValueError):
                timed.resolve_step_tuning(configuration(step=step))
        for method in (*timed.FIXED_DCS_METHODS, timed.ROTATION_METHOD, 'unknown'):
            config = configuration()
            config.update(algorithm=method, step_tuning_policies={method: timed.FIXED_SCS_TUNING})
            with self.subTest(method=method), self.assertRaises(ValueError):
                timed.resolve_step_tuning(config)
        for warmup in (0, -1, True, 1.5):
            config = configuration()
            config['warmup'] = warmup
            with self.subTest(warmup=warmup), self.assertRaises(ValueError):
                timed.resolve_step_tuning(config)

    def test_supplied_step_and_ceiling_use_the_actual_scs_scale(self):
        for step in (.025, .05, .1, .2):
            config = configuration(step=step)
            self.assertEqual(timed._initial_proposal_scale(config), step)
            config['step_scale_ceilings'] = {'SCS': step}
            self.assertEqual(timed.resolve_step_scale_ceiling(config), step)
            config['step_scale_ceilings']['SCS'] = step/2
            with self.assertRaises(ValueError):
                timed.resolve_step_scale_ceiling(config)

    def test_fixed_warmup_and_production_exactly_match_author_scan_at_each_scale(self):
        timed.scs_paper._author()
        import jax
        import jax.numpy as jnp
        for dimension in (100, 500):
            mapping = SphereMap(dimension, scale=2., observer=[.08]+[0.]*(dimension-1)).parameters()
            for step in (.025, .05, .1, .2):
                config = configuration(dimension, step)
                block, project, initial, _ = timed._components(mapping, config['target'], 'SCS')
                burn_key, key = jax.random.split(jax.random.key(192))
                states, accepts = jax.jit(lambda state, rng: block(state, rng, 16, jnp.asarray(step)))(initial, burn_key)
                with patch.object(timed, '_calibrate', side_effect=AssertionError('No acceptance adaptation')), \
                        patch('dcs_bench.radial_tuning.calibrate_radial', side_effect=AssertionError('No radial pilot')):
                    backend = timed._Matched(mapping, config, 192)
                np.testing.assert_array_equal(backend.state, states[-1])
                np.testing.assert_array_equal(jax.random.key_data(backend.key), jax.random.key_data(key))
                self.assertEqual(backend.warmup_acceptance, float(np.sum(accepts))/16)
                self.assertEqual(backend.scale, step)
                _, subkey = jax.random.split(key)
                expected = jax.jit(lambda state, rng: timed._radial_batch(
                    block, project, state, rng, jnp.asarray(step), 1, 50.))(states[-1], subkey)
                prefixes, norms = backend.batch()
                np.testing.assert_array_equal(backend.state, expected[0])
                np.testing.assert_array_equal(prefixes, expected[1])
                np.testing.assert_array_equal(norms, expected[2])

    def test_timed_records_fixed_step_and_keeps_rejected_repeats(self):
        for dimension in (100, 500):
            for step in (.025, .05, .1, .2):
                config = configuration(dimension, step)
                traces, record = timed.run_timed(SphereMap(dimension, scale=2.).parameters(), config, 192)
                self.assertEqual(record['tuning_policy'], timed.FIXED_SCS_TUNING)
                for field in ('initial_scale', 'scale', 'final_scale', 'frozen_scale', 'fixed_step', 'scs_fixed_move'):
                    self.assertEqual(record[field], step)
                self.assertIsNone(record['target_acceptance'])
                self.assertFalse(record['step_size_adapted'])
                self.assertFalse(record['adaptation_during_sampling'])
                self.assertEqual(record['warmup_iterations'], 16)
                self.assertEqual(record['tuning_pilot_iterations'], 16)
                self.assertEqual(record['discarded_iterations'], record['iterations']//3)
                self.assertEqual(len(traces), record['estimation_iterations'])
                self.assertIn('Supplied fixed SCS', record['step_policy'])
                self.assertNotIn('radial_tuning_history', record)
                self.assertTrue(np.any(np.diff(traces[:, 1]) == 0.))
                np.testing.assert_allclose(np.mean(traces, axis=0), record['mean_all_iterations'], rtol=2e-13, atol=1e-13)

    def test_author_point_one_policy_ignores_optional_supplied_control(self):
        mapping = SphereMap(100, scale=2.).parameters()
        config = configuration(step=.2)
        config['step_tuning_policies'] = {'SCS': timed.SCS_FIXED_TUNING}
        original = dict(config)
        original.pop('scs_fixed_move')
        expected = timed._Matched(mapping, original, 192)
        actual = timed._Matched(mapping, config, 192)
        np.testing.assert_array_equal(actual.state, expected.state)
        self.assertEqual(actual.scale, .1)
        self.assertEqual(actual.tuning_metadata, expected.tuning_metadata)
        for a, b in zip(actual.batch(), expected.batch()):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
