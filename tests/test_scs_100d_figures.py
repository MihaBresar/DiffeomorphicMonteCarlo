"""Check that the report preserves the actual estimators and protocol."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.scs_100d_figures import render_report


def example_summary():
    truth = np.array([.012345678901234567, 13.785260238724735])
    cases = {}
    for index, name in enumerate(('DCS-RWM', 'DCS-BW', 'SCS')):
        means = truth+np.array([[.002, .5], [-.001, -.8], [.0003, .15]])*(index+1)
        records = []
        for chain, mean in enumerate(means):
            records.append({'chain': chain, 'iterations': 1000, 'seconds': 1.123456789012345+chain,
                            'final_scale': .1+index*.03, 'seed': 12001+chain,
                            'checkpoints': [{'iterations': count, 'mean_all_iterations':
                                             (truth+(mean-truth)*factor).tolist()}
                                            for count, factor in ((50, 2.), (200, 1.5), (500, 1.2), (1000, 1.))]})
        elapsed = float(np.mean([row['seconds'] for row in records]))
        cases[name] = {'chains': 3, 'chain_means': means.tolist(),
                       'mse': np.mean((means-truth)**2, axis=0).tolist(),
                       'bias_squared': ((means.mean(axis=0)-truth)**2).tolist(),
                       'variance_ddof0': np.var(means, axis=0, ddof=0).tolist(),
                       'mean_seconds': elapsed, 'median_seconds': elapsed,
                       'iterations_per_second': 1000/elapsed, 'total_iterations': 3000,
                       'mean_acceptance': .2345678901234567, 'final_scales': [.1+index*.03]*3,
                       'records': records}
    return {'config': {'runs': 3, 'iterations': 1000, 'warmup': 100, 'thin': 10, 'preset': 'quick'},
            'truth': truth.tolist(), 'cases': cases,
            'fitting': {'ball': {'seconds': 2.234567890123456, 'steps': 1001, 'samples_per_step': 2000,
                                  'training_batch_policy': 'fresh'},
                        'sphere': {'seconds': 1.9876543210987654, 'steps': 1000, 'training_draws': 2000,
                                   'total_training_draws': 2000, 'batch_policy': 'fixed'}},
            'maps': {'ball': {'dimension': 100, 'scale': 1.2345678901234567, 'mu': [.01, -.02],
                              'delta': [.03, .04], 'beta': 2.},
                     'sphere': {'scale': 1.0987654321098765, 'observer': [.05, -.06], 'latitude': 1.1}}}


class SCS100DFiguresTests(unittest.TestCase):
    def test_report_preserves_exact_values_counts_and_protocol(self):
        summary = example_summary()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = render_report(root, summary)
            self.assertEqual(artifacts['figures'], ['VsSCS', 'scs_100d_estimates', 'scs_100d_runtime',
                                                   'scs_100d_bias_variance', 'scs_100d_prefix_mse'])
            for name in artifacts['figures']:
                for suffix in ('.png', '.pdf'):
                    self.assertGreater((root/'figures'/(name+suffix)).stat().st_size, 1000)
            with (root/'scs_100d_values.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['method'] for row in rows], list(summary['cases']))
            self.assertEqual((root/'paper_values.csv').read_bytes(), (root/'scs_100d_values.csv').read_bytes())
            with (root/'sample_counts.csv').open(newline='', encoding='utf-8') as stream:
                counts = list(csv.DictReader(stream))
            self.assertEqual([int(row['total_iterations']) for row in counts], [3000]*3)
            for row in rows:
                case = summary['cases'][row['method']]
                self.assertEqual(int(row['iterations_per_chain']), 1000)
                self.assertEqual(int(row['total_iterations']), 3000)
                self.assertEqual(int(row['retained_per_chain']), 100)
                self.assertEqual(float(row['mean_seconds']), case['mean_seconds'])
                for metric, key in enumerate(('tail_probability', 'mean_norm')):
                    self.assertEqual(row[key+'_mse'], repr(case['mse'][metric]))
                    self.assertEqual(float(row[key+'_mse']), case['mse'][metric])
                    estimates = np.array(case['chain_means'])[:, metric]
                    expected = np.mean((estimates-summary['truth'][metric])**2)
                    self.assertEqual(float(row[key+'_mse']), expected)
            with (root/'scs_100d_chains.csv').open(newline='', encoding='utf-8') as stream:
                chains = list(csv.DictReader(stream))
            self.assertEqual(len(chains), 9)
            for row in chains:
                mean = summary['cases'][row['method']]['chain_means'][int(row['chain'])]
                self.assertEqual(float(row['mean_norm_estimate']), mean[1])
                self.assertEqual(float(row['tail_probability_estimate']), mean[0])
            with (root/'scs_100d_prefix_mse.csv').open(newline='', encoding='utf-8') as stream:
                prefix = list(csv.DictReader(stream))
            self.assertEqual(len(prefix), 24)
            for row in prefix:
                if int(row['iterations_per_chain']) == 1000:
                    index = ('tail_probability', 'mean_norm').index(row['metric'])
                    self.assertEqual(float(row['mse']), summary['cases'][row['method']]['mse'][index])
            with (root/'tuned_parameters.csv').open(newline='', encoding='utf-8') as stream:
                parameters = list(csv.DictReader(stream))
            ball_scale = next(row for row in parameters if row['map'] == 'ball' and row['parameter'] == 'scale')
            self.assertEqual(float(ball_scale['value']), summary['maps']['ball']['scale'])
            with (root/'setup_times.csv').open(newline='', encoding='utf-8') as stream:
                setup = {row['map']: row for row in csv.DictReader(stream)}
            self.assertEqual(int(setup['ball']['total_training_draws']), 2002000)
            self.assertEqual(int(setup['sphere']['total_training_draws']), 2000)
            self.assertEqual(float(setup['ball']['fitting_seconds']), summary['fitting']['ball']['seconds'])
            page = (root/'paper_values.html').read_text(encoding='utf-8')
            self.assertIn('Quick execution check only', page)
            self.assertIn('adaptation continues afterward', page)
            self.assertIn('fixed proposal step h = 0.1', page)
            self.assertIn('not a literal reproduction of the SCS publication target', page)
            self.assertIn('equal production iteration counts', page)

    def test_invalid_decomposition_is_rejected(self):
        summary = example_summary()
        summary['cases']['DCS-BW']['variance_ddof0'][1] *= 2
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'MSE must equal'):
                render_report(directory, summary)


if __name__ == '__main__':
    unittest.main()
