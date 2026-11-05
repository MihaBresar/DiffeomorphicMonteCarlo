"""Equal VI work and a fixed reference cloud for both current fitted families."""
from contextlib import redirect_stdout
from io import StringIO
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from dcs_bench import tuning
from dcs_bench.targets import Target
from scripts import reproduce_comparison as suite, reproduce_scs_100d as runner


class MatchedVITests(unittest.TestCase):
    def test_all_presets_backends_and_dimensions_share_work_and_fixed_cloud_policy(self):
        for preset, updates, draws in (('quick', 20, 128), ('reproduce', 400, 500), ('paper', 1000, 2000)):
            for backend in ('jax', 'numba'):
                for target in ('skew-t-100d', 'skew-t-500d'):
                    config = suite.configuration(preset, comparison=backend, target=target)
                    self.assertEqual(config['optimizer_steps'], updates)
                    self.assertEqual(config['optimizer_samples'], draws)
                    self.assertEqual(config['scs_vi'], dict(optimizer_steps=updates, optimizer_samples=draws))
                    self.assertEqual(config['vi_batch_policy'], 'fixed')
                    self.assertEqual(tuning.validate_matched_vi(config), config['scs_vi'])

    def test_real_fits_report_equal_work_and_dcs_draws_its_cloud_only_once(self):
        config = suite.configuration('quick', overrides={'optimizer_steps': 3, 'optimizer_samples': 16})
        config['target'] = {'kind': 'skew_t', 'dimension': 3, 'nu': 3., 'alpha': {'0': 2.}}
        with patch.object(tuning, 'uniform_ball', wraps=tuning.uniform_ball) as draw_cloud, redirect_stdout(StringIO()):
            maps, fitting = runner.prepare(config)
        draw_cloud.assert_called_once()
        for family in ('ball', 'sphere'):
            info = fitting[family]
            self.assertEqual(info['steps'], 3)
            self.assertEqual(info['gradient_updates'], 3)
            self.assertEqual(info['training_draws'], 16)
            self.assertEqual(info['unique_reference_draws'], 16)
            self.assertEqual(info['total_training_draws'], 16)
            self.assertEqual(info['objective_point_evaluations'], 48)
            self.assertEqual(info['batch_policy'], 'fixed')
            self.assertTrue(np.isfinite(maps[family]['scale']))
        self.assertEqual(fitting['ball']['seed'], 42)
        self.assertEqual(fitting['sphere']['seed'], 0)

    def test_mismatched_or_fresh_settings_fail_before_fitting_or_creating_output(self):
        for change in ({'scs_vi': {'optimizer_steps': 3, 'optimizer_samples': 128}},
                       {'vi_batch_policy': 'fresh'}):
            config = suite.configuration('quick')
            config.update(change)
            with self.subTest(change=change), patch.object(runner, 'fit_map') as fit, self.assertRaisesRegex(ValueError, 'Matched VI'):
                runner.prepare(config)
            fit.assert_not_called()
            with tempfile.TemporaryDirectory() as parent:
                output = Path(parent)/'run'
                with self.assertRaisesRegex(ValueError, 'Matched VI'):
                    runner.run(config, output)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
