"""ESS belongs to the main DCS-SCS comparisons, not standard experiments."""
import unittest

from dcs_bench.engine import experiment_names, load_config
from dcs_bench.reported_study import configuration


class StandardDiagnosticPolicyTests(unittest.TestCase):
    def test_every_standard_preset_disables_chain_diagnostics(self):
        for name in experiment_names():
            for preset in ('quick', 'reproduce', 'paper'):
                with self.subTest(name=name, preset=preset):
                    config = load_config(name, preset)
                    self.assertIs(config['ess_enabled'], False)
                    self.assertIs(config['diagnostics_enabled'], False)

    def test_standard_overrides_cannot_restore_the_removed_diagnostics(self):
        for field in ('ess_enabled', 'diagnostics_enabled'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'only in the main DCS-SCS'):
                load_config('eight-schools', overrides={field: True})

    def test_main_jax_and_numba_comparisons_keep_functional_ess(self):
        for backend in ('jax', 'numba'):
            for target in ('skew-t-100d', 'skew-t-500d'):
                with self.subTest(backend=backend, target=target):
                    self.assertTrue(configuration(comparison=backend, target=target)['ess_enabled'])


if __name__ == '__main__':
    unittest.main()
