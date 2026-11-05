"""Keep the comparison's SCS map fit faithful to the pinned author routine."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np

from dcs_bench import scs_paper
from dcs_bench.transforms import BallMap, SphereMap
from scripts import reproduce_scs_100d as runner


class AuthorVIComparisonTests(unittest.TestCase):
    def configuration(self):
        config = runner.configuration('quick', overrides={
            'optimizer_steps': 7, 'optimizer_samples': 19})
        config.update(sphere_map_policy='author-paper-vi',
                      scs_vi={'optimizer_steps': 3, 'optimizer_samples': 16})
        config['target'] = {'kind': 'skew_t', 'dimension': 5, 'nu': 3.,
                            'alpha': {'0': 3., '1': -3.}}
        return config

    @staticmethod
    def ball_fit(target, config, family, seed):
        return BallMap(target.d, beta=config['beta']), {
            'seconds': 1.25, 'steps': config['optimizer_steps'],
            'training_draws': config['optimizer_samples'], 'seed': seed}

    @staticmethod
    def author_result():
        return SphereMap(5, scale=2.).parameters(), {
            'seconds': 2.5, 'steps': 3, 'training_draws': 16,
            'total_training_draws': 16, 'seed': 0, 'learning_rate': .01,
            'batch_policy': 'fixed', 'vi_coordinate_clip': 1000.,
            'gradient_clipping': False, 'loss': [3., 2., 1.],
            'source_commit': scs_paper.UPSTREAM_COMMIT,
            'map_policy': 'SCS-paper-VI'}

    def test_dispatch_preserves_dcs_fit_and_records_actual_author_metadata(self):
        config = self.configuration()
        original = deepcopy(config)
        author_mapping, author_metadata = self.author_result()
        with patch.object(runner, 'fit_map', side_effect=self.ball_fit) as dcs_fit, \
                patch.object(scs_paper, 'fit', return_value=(author_mapping, author_metadata)) as author_fit, \
                redirect_stdout(StringIO()) as progress:
            maps, fitting = runner.prepare(config)

        self.assertEqual(config, original)
        self.assertEqual(dcs_fit.call_count, 1)
        target, settings, family, seed = dcs_fit.call_args.args
        self.assertEqual((target.d, family, seed), (5, 'ball', 42))
        self.assertIs(settings, config)
        author_fit.assert_called_once_with({'target': config['target'], **config['scs_vi']})
        self.assertEqual(maps['sphere'], author_mapping)
        self.assertEqual(fitting['ball']['steps'], 7)
        self.assertEqual(fitting['ball']['training_draws'], 19)
        self.assertEqual(fitting['ball']['seconds'], 1.25)
        for key, value in author_metadata.items():
            self.assertEqual(fitting['sphere'][key], value)
        self.assertTrue(fitting['sphere']['vi_enabled'])
        self.assertTrue(fitting['sphere']['coordinate_clipping'])
        self.assertFalse(fitting['sphere']['cdf_clipping'])
        self.assertNotIn('gradient_norm_cap', fitting['sphere'])
        self.assertNotIn('shared', progress.getvalue().lower())
        for details in fitting.values():
            self.assertNotIn('shared', details['fitting_policy'].lower())

    def test_missing_author_budget_uses_full_author_defaults_not_dcs_budget(self):
        config = self.configuration()
        del config['scs_vi']
        with patch.object(runner, 'fit_map', side_effect=self.ball_fit), \
                patch.object(scs_paper, 'fit', return_value=self.author_result()) as author_fit, \
                redirect_stdout(StringIO()):
            runner.prepare(config)
        author_fit.assert_called_once_with({'target': config['target'],
                                           'optimizer_steps': 1000, 'optimizer_samples': 2000})

    def test_invalid_author_budget_fails_before_either_fit(self):
        for budget in (None, [], {'seed': 42}, {'optimizer_steps': 0},
                       {'optimizer_samples': True}, {'optimizer_steps': 1.5}):
            config = self.configuration()
            config['scs_vi'] = budget
            with self.subTest(budget=budget), patch.object(runner, 'fit_map') as dcs_fit, \
                    patch.object(scs_paper, 'fit') as author_fit, self.assertRaisesRegex(ValueError, 'scs_vi'):
                runner.prepare(config)
            dcs_fit.assert_not_called()
            author_fit.assert_not_called()

    def test_vi_off_keeps_both_maps_untuned_without_calling_either_fit(self):
        config = self.configuration()
        config['vi'] = 'off'
        with patch.object(runner, 'fit_map') as dcs_fit, \
                patch.object(scs_paper, 'fit') as author_fit, redirect_stdout(StringIO()):
            maps, fitting = runner.prepare(config)
        dcs_fit.assert_not_called()
        author_fit.assert_not_called()
        for family in ('ball', 'sphere'):
            self.assertEqual(maps[family]['scale'], 1.)
            np.testing.assert_array_equal(maps[family]['mu'], 0.)
            np.testing.assert_array_equal(maps[family]['delta' if family == 'ball' else 'observer'], 0.)
            self.assertEqual(fitting[family]['seconds'], 0.)
            self.assertEqual(fitting[family]['steps'], 0)
            self.assertFalse(fitting[family]['vi_enabled'])

    def test_missing_or_old_policy_cannot_restore_in_house_scs_fit(self):
        for policy in (None, 'vi', 'sphere-vi'):
            config = self.configuration()
            if policy is None:
                config.pop('sphere_map_policy')
            else:
                config['sphere_map_policy'] = policy
            with self.subTest(policy=policy), \
                    patch.object(runner, 'fit_map', side_effect=self.ball_fit) as dcs_fit, \
                    patch.object(scs_paper, 'fit', return_value=self.author_result()) as author_fit, \
                    redirect_stdout(StringIO()):
                _, fitting = runner.prepare(config)
            self.assertEqual([call.args[2] for call in dcs_fit.call_args_list], ['ball'])
            author_fit.assert_called_once_with({'target': config['target'], **config['scs_vi']})
            self.assertEqual(fitting['sphere']['seed'], 0)
            self.assertEqual(fitting['sphere']['map_policy'], 'SCS-paper-VI')

    def test_small_real_author_fit_matches_adapter_and_preserves_fit_cost(self):
        config = self.configuration()
        with patch.object(runner, 'fit_map', side_effect=self.ball_fit), redirect_stdout(StringIO()):
            maps, fitting = runner.prepare(config)
            expected, expected_metadata = scs_paper.fit({'target': config['target'], **config['scs_vi']})
        for key in ('mu', 'observer', 'scale'):
            np.testing.assert_array_equal(maps['sphere'][key], expected[key])
        np.testing.assert_array_equal(fitting['sphere']['loss'], expected_metadata['loss'])
        self.assertEqual(fitting['sphere']['seed'], 0)
        self.assertEqual(fitting['sphere']['batch_policy'], 'fixed')
        self.assertEqual(fitting['sphere']['total_training_draws'], 16)
        self.assertEqual(fitting['sphere']['vi_coordinate_clip'], 1000.)
        self.assertFalse(fitting['sphere']['gradient_clipping'])
        self.assertGreater(fitting['sphere']['seconds'], 0.)
        self.assertEqual(fitting['sphere']['source_commit'], scs_paper.UPSTREAM_COMMIT)


if __name__ == '__main__':
    unittest.main()
