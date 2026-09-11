"""Ensure discarded transient states cannot leak into benchmark estimates."""
import unittest
import numpy as np
from dcs_bench.burnin import discard_burnin
from dcs_bench.engine import experiment_names, load_config
from dcs_bench.samplers import run_metropolis
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap


class BurninTests(unittest.TestCase):
    def test_transient_is_removed_using_original_iteration_grid(self):
        # Seven early transient iterations; saved states at 3,6,9,...,21.
        values = np.array([999., 999., 1., 2., 3., 4., 5.])[:, None]
        record = dict(iterations=21, thin=3, warmup_iterations=100)
        draws, metadata = discard_burnin(values, record, {'burnin_fraction': 1/3})
        self.assertEqual(float(draws.mean()), 3.)
        self.assertEqual(metadata['tuning_iterations'], 100)
        self.assertEqual(metadata['burnin_iterations'], 7)
        self.assertEqual(metadata['first_retained_iteration'], 9)
        self.assertEqual(metadata['estimation_iterations'], 14)
        again, repeated = discard_burnin(draws, metadata, {'burnin_fraction': 1/3})
        np.testing.assert_array_equal(again, draws)
        self.assertEqual(repeated, metadata)

    def test_full_paper_boundary_and_actual_timed_count(self):
        config = dict(burnin_fraction=1/3, iterations=500000, time_budget=20.)
        for count, thin, discarded, used, saved in [(500000, 10, 166667, 333333, 33334),
                                                  (768, 7, 256, 512, 73)]:
            with self.subTest(count=count):
                values=np.zeros((count//thin, 2))
                draws, record=discard_burnin(values, dict(iterations=count, thin=thin), config)
                self.assertEqual(record['burnin_iterations'], discarded)
                self.assertEqual(record['estimation_iterations'], used)
                self.assertEqual(len(draws), saved)

    def test_all_current_engine_configs_enable_rule(self):
        for name in experiment_names():
            with self.subTest(name=name):
                self.assertEqual(load_config(name, 'quick')['burnin_fraction'], 1/3)

    def test_sampling_path_discards_once_and_freezes_tuning(self):
        target=Target(dict(kind='skew_t', dimension=2, nu=3, alpha={}))
        config=dict(warmup=50, iterations=90, thin=2)
        mapping=BallMap(2)
        full, raw=run_metropolis(target, 'DCS-BW', mapping, config, 81)
        kept, record=run_metropolis(target, 'DCS-BW', mapping, {**config, 'burnin_fraction':1/3}, 81)
        np.testing.assert_array_equal(kept, full[15:])
        self.assertEqual(record['iterations'], 90)
        self.assertEqual(record['burnin_iterations'], 30)
        self.assertEqual(record['retained_draws'], 30)
        self.assertFalse(record['adaptation_during_sampling'])
        self.assertEqual(record['frozen_scale'], raw['frozen_scale'])

    def test_invalid_saved_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'iteration indices'):
            discard_burnin(np.zeros((9,2)), dict(iterations=30,thin=2), {'burnin_fraction':1/3})


if __name__ == '__main__':
    unittest.main()
