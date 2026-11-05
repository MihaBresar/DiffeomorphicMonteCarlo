"""All SCS fitting paths use the pinned author's optimizer on the actual target."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from dcs_bench import scs_paper
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap
from dcs_bench.tuning import fit_map


class UniversalSCSFitTests(unittest.TestCase):
    def test_author_budget_is_independent_of_dcs_settings_and_validated(self):
        self.assertEqual(scs_paper.author_vi_budget({
            'optimizer_steps': 4, 'optimizer_samples': 12, 'fit_seed': 42}),
            {'optimizer_steps': 1000, 'optimizer_samples': 2000})
        self.assertEqual(scs_paper.author_vi_budget({'scs_vi': {'optimizer_steps': 3}}),
            {'optimizer_steps': 3, 'optimizer_samples': 2000})
        for value in (None, [], {'seed': 42}, {'optimizer_steps': 0},
                      {'optimizer_samples': True}, {'optimizer_steps': 1.5}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'scs_vi'):
                scs_paper.author_vi_budget({'scs_vi': value})

    def test_logistic_and_skew_t_fits_match_direct_author_optimizer(self):
        import jax
        specs = [
            {'kind': 'logistic', 'dimension': 3, 'nu': 4., 'prior_scale': 2.5,
             'n_observations': 12, 'data_seed': 42},
            {'kind': 'skew_t', 'dimension': 3, 'nu': .5, 'alpha': {}},
        ]
        for spec in specs:
            with self.subTest(kind=spec['kind']):
                target = Target(spec)
                config = {'scs_vi': {'optimizer_steps': 3, 'optimizer_samples': 16}}
                actual, metadata = fit_map(target, config, 'sphere', 981)
                model = scs_paper._author().SCP(target.d, 1.1)
                params, losses = model.minimize_reverse_kl(
                    lambda x: target.log_prob(x, 'jax'), seed=0, ntrain=16,
                    learning_rate=.01, max_iter=3, clip_value=1000.)
                jax.block_until_ready(params)
                observer, centre, radius = model.transform_params(params)
                np.testing.assert_array_equal(actual.observer, observer)
                np.testing.assert_array_equal(actual.mu, centre)
                self.assertEqual(actual.scale, float(radius))
                np.testing.assert_array_equal(metadata['loss'], losses)
                self.assertEqual(metadata['seed'], 0)
                self.assertEqual(metadata['total_training_draws'], 16)
                self.assertEqual(metadata['batch_policy'], 'fixed')
                self.assertEqual(metadata['target_kind'], spec['kind'])
                self.assertFalse(metadata['gradient_clipping'])
                self.assertFalse(metadata['symmetric_constraints'])
                self.assertEqual(metadata['scale_initialization'], 'unit')
                self.assertTrue(metadata['coordinate_clipping'])
                self.assertEqual(metadata['source_commit'], scs_paper.UPSTREAM_COMMIT)
                self.assertIn('src/scp_core.py', metadata['source_sha256'])
                if spec['kind'] == 'logistic':
                    self.assertNotIn('target_skew_cdf', metadata)

    def test_old_sphere_settings_and_dcs_seed_cannot_select_another_fitter(self):
        target = Target({'kind': 'skew_t', 'dimension': 3, 'nu': 3., 'alpha': {'0': 2.}})
        config = {'beta': .2, 'latitude': 1.1, 'ball_map_policy': 'vanilla',
            'sphere_map_policy': 'vi', 'vi_batch_policy': 'fresh',
            'map_scale_initialization': 'radial_median', 'optimizer_steps': 999,
            'optimizer_samples': 999, 'scs_vi': {'optimizer_steps': 3, 'optimizer_samples': 16}}
        original = deepcopy(config)
        with patch('dcs_bench.tuning.uniform_cap', side_effect=AssertionError('Old SCS optimizer used')):
            first, first_meta = fit_map(target, config, 'sphere', 581)
            second, second_meta = fit_map(target, config, 'sphere', 96001)
        self.assertEqual(config, original)
        self.assertEqual(first.parameters(), second.parameters())
        self.assertEqual(first_meta['loss'], second_meta['loss'])
        self.assertEqual(first_meta['steps'], 3)
        self.assertEqual(first_meta['training_draws'], 16)

    def test_vi_off_keeps_sphere_untuned_and_never_calls_optimizer(self):
        target = Target({'kind': 'skew_t', 'dimension': 3, 'nu': 3., 'alpha': {}})
        for disabled in ({'vi': 'off'}, {'sphere_map_policy': 'vanilla'}):
            with self.subTest(disabled=disabled), patch.object(scs_paper, 'fit') as fit:
                mapping, metadata = fit_map(target, disabled, 'sphere', 123)
            fit.assert_not_called()
            self.assertEqual(mapping.parameters(), SphereMap(3).parameters())
            self.assertFalse(metadata['vi_enabled'])
            self.assertEqual(metadata['total_training_draws'], 0)

    def test_dcs_identity_and_vanilla_paths_do_not_use_author_fit(self):
        target = Target({'kind': 'skew_t', 'dimension': 3, 'nu': 3., 'alpha': {}})
        with patch.object(scs_paper, 'fit') as fit:
            mapping, metadata = fit_map(target, {'beta': .2, 'ball_map_policy': 'vanilla'}, 'ball', 581)
        fit.assert_not_called()
        self.assertEqual(mapping.parameters(), BallMap(3, beta=.2).parameters())
        self.assertEqual(metadata['map_policy'], 'vanilla')

    def test_latitude_is_preserved_and_invalid_values_rejected_before_fit(self):
        spec = {'kind': 'skew_t', 'dimension': 3, 'nu': 3., 'alpha': {}}
        mapping, metadata = scs_paper.fit({'target': spec, 'latitude': 1.3,
                                         'optimizer_steps': 2, 'optimizer_samples': 16})
        self.assertEqual(mapping['latitude'], 1.3)
        self.assertEqual(metadata['map_policy'], 'SCS-paper-VI')
        for latitude in (True, None, '1.1', 0.9, 2., float('nan'), float('inf')):
            with self.subTest(latitude=latitude), patch.object(scs_paper, '_author') as author:
                with self.assertRaisesRegex(ValueError, 'latitude'):
                    scs_paper.fit({'target': spec, 'latitude': latitude})
            author.assert_not_called()


if __name__ == '__main__':
    unittest.main()
