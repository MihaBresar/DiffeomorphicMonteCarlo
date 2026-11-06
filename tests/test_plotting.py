"""Verify restored outputs and exact numerical exports from stored checkpoints."""
import run_experiment
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from dcs_bench.engine import save_chain, write_json
from tests.configuration_fixtures import load_config
from dcs_bench.references import make_reference, functional_draws
from dcs_bench.targets import Target
from dcs_bench.plotting import rebuild, render_saved_summary, burnin_counts, number, method_labels, ball_map_description, skew_figures, LABELS


class FigureTests(unittest.TestCase):
    def test_plot_only_reuses_summary_and_explicit_recompute_reads_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/'summary.json', {'scenarios': []})
            with patch('dcs_bench.plotting.render_saved_summary') as render, \
                 patch('dcs_bench.plotting.rebuild') as recompute:
                self.assertEqual(run_experiment.main(['--plot-only', str(root)]), 0)
                render.assert_called_once_with(root)
                recompute.assert_not_called()
            with patch('dcs_bench.plotting.render_saved_summary') as render, \
                 patch('dcs_bench.plotting.rebuild') as recompute:
                self.assertEqual(run_experiment.main(['--plot-only', str(root), '--recompute-diagnostics']), 0)
                recompute.assert_called_once_with(root)
                render.assert_not_called()
            (root/'summary.json').unlink()
            with patch('dcs_bench.plotting.rebuild') as recompute:
                self.assertEqual(run_experiment.main(['--plot-only', str(root)]), 0)
                recompute.assert_called_once_with(root)

    def test_sampling_burnin_counts_remain_separate_from_generated_counts(self):
        records = [{'iterations': 1000, 'burnin_iterations': 333, 'estimation_iterations': 667},
                   {'iterations': 1200, 'burnin_iterations': 400, 'estimation_iterations': 800}]
        result = burnin_counts(records)
        self.assertEqual(result['total_burnin_iterations'], 733)
        self.assertEqual(result['total_estimation_iterations'], 1467)
        self.assertEqual(result['mean_estimation_iterations'], 733.5)
        self.assertEqual(burnin_counts([{'iterations': 1000}]), {})
        records[0]['estimation_iterations'] = 1000
        with self.assertRaisesRegex(ValueError, 'must sum'):
            burnin_counts(records)

    def test_variant_labels_and_historical_scale_description(self):
        config = load_config('skew-t-100d-fixed', 'quick', {'ball_map_policy': 'vanilla'})
        labels = method_labels(['DCS-BW', 'SCS'], config)
        self.assertIn('vanilla', labels[0])
        self.assertNotIn('vanilla', labels[1])
        self.assertIn('R=1, centre=0, Möbius=0', ball_map_description(config))
        legacy = dict(config, ball_scale_policy='sqrt-d')
        legacy.pop('ball_map_policy')
        self.assertIn('R=√d', method_labels(['DCS-BW'], legacy)[0])
        self.assertIn('historical', ball_map_description(legacy))
        posterior = load_config('eight-schools', 'quick', {'ball_map_policy': 'vanilla'})
        self.assertEqual(ball_map_description(posterior), 'posterior identity map')
        self.assertNotIn('vanilla', method_labels(['DCS-BW'], posterior)[0])

    def test_nuts_performance_figure_uses_counts_instead_of_ess(self):
        config = load_config('skew-t-200d-nuts', 'quick')
        target = Target(config['targets'][0])
        ref = {'functional_names': ['tail_probability', 'mean_norm'],
               'functional_truth': [.01, 10.]}
        data = {algorithm: {'estimates': np.array([[.02, 9.], [.01, 10.]]),
            'records': [{'iterations': 100, 'retained_draws': 50}],
            'summary': {'mse': [.001, 1.], 'mean_iterations': 100,
                        'mean_retained_draws': 50, 'mean_sampling_seconds': 2.}}
                for algorithm in config['algorithms']}
        saved = []
        def capture(fig, folder, name, quick):
            saved.append((name, [ax.get_title() for ax in fig.axes]))
            import matplotlib.pyplot as plt
            plt.close(fig)
        with tempfile.TemporaryDirectory() as tmp, \
             patch('dcs_bench.plotting.skew_qq_figure'), \
             patch('dcs_bench.plotting.save_figure', side_effect=capture):
            skew_figures(target, ref, data, config, LABELS, Path(tmp), False)
        performance = next(titles for name, titles in saved if name.endswith('_mse_and_sampling'))
        self.assertEqual(performance[-1], 'Average Iterations Generated')
        self.assertFalse(any('ESS' in title for _, titles in saved for title in titles))

    def test_posterior_reports_skip_unused_chain_quantiles_and_preserve_rmse(self):
        from dcs_bench.processing import process_chains
        for experiment in ('eight-schools', 'garch11'):
            with self.subTest(experiment=experiment), tempfile.TemporaryDirectory() as tmp:
                config = load_config(experiment, 'quick')
                target = Target(config['targets'][0])
                root = Path(tmp)
                write_json(root/'configuration.json', config)
                write_json(root/'manifest.json', {'fingerprint': 'posterior-estimates-only'})
                scenario = root/'scenario-0'
                scenario.mkdir()
                truth = np.r_[np.zeros(target.d), np.ones(target.d)]
                reference = {'functional_names': [f'mean_{name}' for name in target.names]
                    + [f'second_{name}' for name in target.names],
                    'functional_truth': truth.tolist(), 'functional_reference_mcse': [0.]*(2*target.d),
                    'quantiles': []}
                write_json(scenario/'reference.json', reference)
                write_json(scenario/'setup.json', {})
                rng = np.random.default_rng(176)
                expected = {}
                for algorithm in config['algorithms']:
                    expected[algorithm] = []
                    for chain in range(config['runs']):
                        draws = rng.normal(size=(41+chain, target.d))
                        expected[algorithm].append(functional_draws(target, draws, config['threshold']).mean(axis=0))
                        save_chain(scenario/'chains'/f'{algorithm}-{chain:03d}.npz', draws,
                            dict(fingerprint='posterior-estimates-only', seed=chain,
                                iterations=len(draws), retained_draws=len(draws), sampling_seconds=1.))
                def no_chain_quantiles(*args, **kwargs):
                    self.assertFalse(kwargs['quantiles_enabled'])
                    with patch('numpy.quantile', side_effect=AssertionError('Unused chain quantiles requested')):
                        return process_chains(*args, **kwargs)
                with patch('dcs_bench.plotting.process_chains', side_effect=no_chain_quantiles):
                    rebuild(root)
                results = json.loads((root/'summary.json').read_text())['scenarios'][0]['algorithms']
                for algorithm, result in results.items():
                    estimates = np.asarray(expected[algorithm])
                    np.testing.assert_array_equal(result['chain_functional_estimates'], estimates)
                    np.testing.assert_array_equal(result['mse'], ((estimates-truth)**2).mean(axis=0))
                    for moment, label in enumerate(('first', 'second')):
                        sl = slice(moment*target.d, (moment+1)*target.d)
                        expected_rmse = np.sqrt(np.mean((estimates[:, sl]-truth[sl])**2))
                        self.assertEqual(result[f'{label}_moment_rmse_across_chains_and_coordinates'], expected_rmse)
                    self.assertEqual(result['chain_quantiles'], [[] for _ in range(config['runs'])])
                    self.assertFalse(result['result_processing']['quantiles_computed'])
                # Empty serialized quantiles remain safe for saved-summary replot.
                with patch('dcs_bench.engine.read_chain', side_effect=AssertionError('No chain reread')):
                    render_saved_summary(root)
                self.assertTrue((root/'posterior_table.tex').exists())

    def test_original_figure_set_and_values_from_unchanged_checkpoints(self):
        config = load_config('skew-t-50d-nu-0.5', 'quick', {'ball_map_policy': 'vanilla'})
        config['targets'][0]['dimension'] = 4
        config['reference_samples'] = 400
        config['reference_batch_size'] = 200
        config['thin'] = 10
        target = Target(config['targets'][0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/'configuration.json', config)
            write_json(root/'manifest.json', {'fingerprint': 'saved-chain-test'})
            scenario = root/'scenario-0'
            ref = make_reference(target, config, scenario)
            write_json(scenario/'setup.json', {})
            write_json(scenario/'DCS-BW_compilation.json', {'seconds': 4.0})
            rng = np.random.default_rng(721)
            expected = {}
            for a in config['algorithms']:
                estimates = []
                for i, n in enumerate([51, 53]):
                    x = rng.standard_t(.5, size=(n, 4))
                    estimates.append(functional_draws(target, x, config['threshold']).mean(axis=0))
                    save_chain(scenario/'chains'/f'{a}-{i:03d}.npz', x,
                        dict(fingerprint='saved-chain-test', seed=i, iterations=n*10,
                             retained_draws=n, sampling_seconds=1.0))
                expected[a] = np.mean((np.array(estimates)-ref['functional_truth'])**2, axis=0)
            before = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in scenario.rglob('*.npz')}
            (root/'functional_ess.csv').write_text('obsolete diagnostic export')
            with patch('dcs_bench.samplers.run_metropolis', side_effect=AssertionError('No MCMC during replot')), \
                 patch('dcs_bench.processing.mcmc_diagnostics', side_effect=AssertionError('No chain diagnostics')), \
                 patch('dcs_bench.processing._FunctionalStore', side_effect=AssertionError('No diagnostic staging')), \
                 patch('dcs_bench.functional_ess.functional_ess', side_effect=AssertionError('No ESS computation')):
                rebuild(root)
            self.assertFalse((root/'functional_ess.csv').exists())
            processing = json.loads((root/'processing.json').read_text())
            self.assertFalse(processing['diagnostics_computed'])
            self.assertFalse(processing['functional_ess_computed'])
            summary = json.loads((root/'summary.json').read_text())['scenarios'][0]['algorithms']
            self.assertEqual(summary['DCS-BW']['mean_seconds_including_setup'], 3.0)
            self.assertEqual(summary['SCS']['mean_seconds_including_setup'], 1.0)
            for result in summary.values():
                for key in ('functional_ess', 'cross_chain_diagnostics', 'independent_chain_diagnostics',
                            'total_coordinate_bulk_ess', 'chain_coordinate_bulk_ess'):
                    self.assertNotIn(key, result)
                self.assertEqual(result['result_processing']['staged_functional_bytes'], 0)
            for p, digest in before.items():
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)
            for suffix in ('', '_marginal_qq', '_ergodic_averages', '_sample_counts'):
                self.assertTrue((root/'figures'/f'SuperInfiniteFirstTest{suffix}.png').exists())
            self.assertTrue((root/'figures/supplementary/SuperInfiniteFirstTest_directional.png').exists())
            with (root/'paper_values.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertFalse(any('ess' in key or 'rhat' in key for key in rows[0]))
            for row in rows:
                self.assertEqual(row['ball_map_policy'], 'vanilla')
                k = ref['functional_names'].index(row['functional'])
                self.assertEqual(float(row['mse']), expected[row['algorithm']][k])
                self.assertEqual(int(row['total_iterations']), 1040)
                self.assertEqual(int(row['total_retained_samples']), 104)
                self.assertEqual(float(row['mean_retained_samples']), 52)
            self.assertIn(number(expected['DCS-BW'][0], scientific=True), (root/'paper_values.html').read_text())
            self.assertIn('vanilla: R=1, centre=0, Möbius=0', (root/'paper_values.html').read_text(encoding='utf-8'))
            numerical = {path: path.read_bytes() for path in root.rglob('*')
                         if path.suffix in ('.json', '.csv', '.npz')}
            with patch('dcs_bench.engine.read_chain', side_effect=AssertionError('No chain reads for label refresh')), \
                 patch('dcs_bench.processing.mcmc_diagnostics', side_effect=AssertionError('No recomputed diagnostics')):
                render_saved_summary(root)
            for path, payload in numerical.items():
                self.assertEqual(path.read_bytes(), payload)


if __name__ == '__main__':
    unittest.main()
