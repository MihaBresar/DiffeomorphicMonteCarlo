"""Standard reports keep estimates and timing without chain diagnostics."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dcs_bench.engine import experiment_names, load_config, save_chain, write_json
from dcs_bench.plotting import rebuild, render_saved_summary
from dcs_bench.radial_reporting import compact_radial_report, summarize_radial
from dcs_bench.references import functional_draws, make_reference, radial_truth
from dcs_bench.targets import Target


class CompactFigureTests(unittest.TestCase):
    def test_only_baselines_default_to_compact_in_every_preset(self):
        for name in experiment_names():
            for preset in ('quick', 'reproduce', 'paper'):
                config = load_config(name, preset)
                self.assertEqual(compact_radial_report(config), name == 'skew-t-200d-baselines')

    def test_radial_reference_needs_no_iid_or_quantile_work(self):
        config = load_config('skew-t-200d-baselines', 'paper')
        target = Target(config['targets'][0])
        with tempfile.TemporaryDirectory() as tmp, \
             patch('dcs_bench.references.skew_t_iid', side_effect=AssertionError('No simulated reference')), \
             patch('numpy.quantile', side_effect=AssertionError('No reference quantiles')):
            ref = make_reference(target, config, Path(tmp))
            names, truth, _ = radial_truth(target, config['threshold'])
            self.assertEqual(ref['functional_names'], names)
            np.testing.assert_array_equal(ref['functional_truth'], truth)
            self.assertEqual(ref['draws'], 0)
            self.assertEqual(ref['functional_reference_mcse'], [0., 0.])
            self.assertFalse((Path(tmp)/'reference.npz').exists())

    def test_cached_report_values_figures_and_legacy_summary_replot(self):
        config = load_config('skew-t-200d-baselines', 'quick')
        config['targets'][0]['dimension'] = 4
        target = Target(config['targets'][0])
        rng = np.random.default_rng(321)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/'configuration.json', config)
            write_json(root/'manifest.json', {'fingerprint': 'compact-check'})
            scenario = root/'scenario-0'
            ref = make_reference(target, config, scenario)
            write_json(scenario/'setup.json', {})
            expected = {}
            for algorithm in config['algorithms']:
                expected[algorithm] = []
                for index, n in enumerate((41, 53)):
                    draws = 20*rng.standard_t(3, size=(n, 4))
                    expected[algorithm].append(functional_draws(target, draws, config['threshold']).mean(axis=0)[:2])
                    save_chain(scenario/'chains'/f'{algorithm}-{index:03d}.npz', draws,
                        dict(fingerprint='compact-check', seed=index, retained_draws=n,
                             iterations=3*n, burnin_iterations=n, estimation_iterations=2*n,
                             sampling_seconds=1.+index, thin=config['thin'],
                             radial_summary=summarize_radial(draws, target, config['threshold'],
                                 sampling_seconds=1.+index, draw_stride=config['thin'], ess_enabled=True)))
            with patch('dcs_bench.engine.read_chain', side_effect=AssertionError('Use cached estimates')), \
                 patch('dcs_bench.plotting.process_chains', side_effect=AssertionError('No full diagnostics')), \
                 patch('dcs_bench.plotting.skew_qq_figure', side_effect=AssertionError('No coordinate quantiles')), \
                 patch('dcs_bench.functional_ess.estimate_ess', side_effect=AssertionError('No ESS computation')):
                rebuild(root)
            saved = json.loads((root/'summary.json').read_text())
            for algorithm, result in saved['scenarios'][0]['algorithms'].items():
                np.testing.assert_allclose(result['chain_functional_estimates'], expected[algorithm], rtol=2e-15)
                errors = (np.asarray(expected[algorithm])-ref['functional_truth'])**2
                np.testing.assert_allclose(result['mse'], errors.mean(axis=0), rtol=1e-14)
                self.assertEqual(result['total_retained_draws'], 94)
                self.assertEqual(result['total_iterations'], 282)
                self.assertEqual(result['total_burnin_iterations'], 94)
                self.assertEqual(result['mean_sampling_seconds'], 1.5)
                self.assertEqual(result['result_processing']['cached_chains'], 2)
                self.assertNotIn('functional_ess', result)
                self.assertFalse(result['result_processing']['functional_ess_computed'])
                for record in result['records']:
                    self.assertNotIn('functional_ess', record)
                    self.assertNotIn('functional_ess', record['radial_summary'])
                for omitted in ('total_coordinate_bulk_ess', 'cross_chain_diagnostics', 'independent_chain_diagnostics'):
                    self.assertNotIn(omitted, result)
            figures = {path.name for path in (root/'figures').iterdir()}
            self.assertEqual(figures, {f'VsRWM_NUTS{suffix}.{extension}'
                for suffix in ('', '_ergodic_averages', '_sample_counts') for extension in ('png', 'pdf')})
            for filename in ('paper_values.csv', 'chain_metrics.csv'):
                with (root/filename).open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual({row['functional'] for row in rows}, set(ref['functional_names']))
                self.assertTrue(all('estimate' in row for row in rows))
                self.assertFalse(any('ess' in key or 'rhat' in key for key in rows[0]))
                self.assertFalse(any('rhat' in key or 'ess_bulk' in key for key in rows[0]))
            with (root/'paper_values.csv').open() as stream:
                for row in csv.DictReader(stream):
                    k = ref['functional_names'].index(row['functional'])
                    self.assertAlmostEqual(float(row['estimate']), np.mean(expected[row['algorithm']], axis=0)[k])
            self.assertFalse(json.loads((root/'processing.json').read_text())['functional_ess_computed'])
            self.assertFalse(json.loads((root/'processing.json').read_text())['diagnostics_computed'])
            self.assertFalse((root/'functional_ess.csv').exists())

            # Historical completed summaries may contain four quantities.
            # Compact replot selects radial values without rewriting that data.
            reference = saved['scenarios'][0]['reference']
            reference['functional_names'] += ['direction_positive_probability', 'mean_tanh_direction']
            reference['functional_truth'] += [.5, 0.]
            reference['functional_reference_mcse'] += [.01, .01]
            for result in saved['scenarios'][0]['algorithms'].values():
                for key in ('mse', 'rmse'):
                    result[key] += [.1, .2]
                for key in ('chain_functional_estimates', 'mse_bootstrap_95_interval'):
                    for row in result[key]:
                        row += [.3, .4]
            write_json(root/'summary.json', saved)
            before = (root/'summary.json').read_bytes()
            with patch('dcs_bench.engine.read_chain', side_effect=AssertionError('No sampler reads')):
                render_saved_summary(root)
            self.assertEqual(before, (root/'summary.json').read_bytes())
            with (root/'paper_values.csv').open() as stream:
                self.assertEqual({row['functional'] for row in csv.DictReader(stream)}, set(ref['functional_names']))


if __name__ == '__main__':
    unittest.main()
