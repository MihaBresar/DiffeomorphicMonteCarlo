"""Author fitting budgets and provenance across the generic entry points."""
import unittest

from dcs_bench.engine import source_hashes
from tests.configuration_fixtures import load_config


class SCSFitConfigurationTests(unittest.TestCase):
    def test_every_scs_configuration_records_separate_author_budgets(self):
        for name in ('logistic-regression', 'skew-t-50d-nu-0.5', 'skew-t-100d-fixed'):
            for preset, steps, draws in (('quick', 20, 128), ('reproduce', 400, 500),
                                          ('paper', 1000, 2000)):
                with self.subTest(name=name, preset=preset):
                    config = load_config(name, preset)
                    self.assertEqual(config['scs_vi'], dict(optimizer_steps=steps, optimizer_samples=draws))
                    self.assertEqual(config['sphere_map_policy'], 'author-paper-vi')
        # Both families now share the author's full computation budget.
        self.assertEqual(load_config('logistic-regression', 'paper')['optimizer_steps'], 1000)

    def test_explicit_budget_overrides_and_invalid_scs_settings(self):
        config = load_config('logistic-regression', 'quick', {'optimizer_steps': 7})
        self.assertEqual(config['scs_vi'], dict(optimizer_steps=7, optimizer_samples=128))
        with self.assertRaisesRegex(ValueError, 'identical'):
            load_config('logistic-regression', 'quick', {
                'optimizer_steps': 7, 'scs_vi': {'optimizer_steps': 3, 'optimizer_samples': 16}})
        for invalid in (None, [], {'seed': 42}, {'optimizer_steps': 0}):
            if invalid is None:
                continue  # None is the CLI's unspecified-value sentinel.
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                load_config('logistic-regression', 'quick', {'scs_vi': invalid})

    def test_source_fingerprint_covers_the_executed_author_optimizer(self):
        hashes = source_hashes()
        for path in ('third_party/scs_paper/src/scp_core.py',
                     'third_party/scs_paper/src/train.py', 'third_party/scs_paper/manifest.json'):
            self.assertIn(path, hashes)
            self.assertEqual(len(hashes[path]), 64)

    def test_non_scs_experiments_do_not_claim_scs_fitting(self):
        for name in ('eight-schools', 'garch11', 'skew-t-200d-nuts', 'skew-t-200d-baselines'):
            config = load_config(name, 'quick')
            self.assertNotIn('scs_vi', config)


if __name__ == '__main__':
    unittest.main()
