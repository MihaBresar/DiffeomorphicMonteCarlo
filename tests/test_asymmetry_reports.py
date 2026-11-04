"""Coordinate measurements supplement saved radial figures without relabeling history."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import START_HERE as launcher
import run_experiment
from scripts.scs_100d_figures import render_report
from tests.test_comparison_ui import comparison_summary


def coordinate_summary(optimized=False):
    summary = comparison_summary(optimized=optimized)
    summary['config'].update(protocol='timed-numba-comparison-v2' if optimized else 'timed-sampler-comparison-v2',
                             coordinate_metrics=True, coordinate_threshold=3.)
    truth = np.r_[summary['truth'], .812345678912345, .024567891234567]
    summary['truth'] = truth.tolist()
    summary['functionals'] = ['tail_probability', 'mean_norm', 'mean_x1', 'x1_tail_probability']
    for index, case in enumerate(summary['cases'].values()):
        deviations = np.array([[.02, .002], [-.03, -.001], [.01, .0003]])*(index+1)
        values = np.column_stack([case['chain_means'], truth[2:]+deviations])
        case.update(chain_means=values.tolist(), mse=np.mean((values-truth)**2, axis=0).tolist(),
                    bias_squared=((values.mean(axis=0)-truth)**2).tolist(),
                    variance_ddof0=values.var(axis=0).tolist())
    summary['reference'] = {'functionals': summary['functionals'], 'truth': truth.tolist(),
                            'reference_mcse': [0., 0., .002, .00005],
                            'exact': [True, True, False, False],
                            'source': 'Independent skew-t reference', 'draws': 1000000, 'seed': 97531}
    return summary


class AsymmetryReportTests(unittest.TestCase):
    def test_coordinate_outputs_keep_radial_pngs_and_exact_values(self):
        summary = coordinate_summary()
        old = comparison_summary()
        old['config']['protocol'] = 'timed-sampler-comparison-v2'
        before = deepcopy(summary)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_report = render_report(root/'radial', old)
            report = render_report(root/'extended', summary)
            self.assertEqual(report['figures'], old_report['figures']+
                             ['coordinate_mean', 'coordinate_tail_probability'])
            for filename in old_report['figures']:
                self.assertEqual((root/'radial/figures'/f'{filename}.png').read_bytes(),
                                 (root/'extended/figures'/f'{filename}.png').read_bytes())
            with (root/'extended/paper_values.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                case = summary['cases'][row['method']]
                for index, metric in enumerate(summary['functionals']):
                    self.assertEqual(float(row[metric+'_mse']), case['mse'][index])
                    self.assertEqual(float(row[metric+'_truth']), summary['truth'][index])
                    self.assertEqual(float(row[metric+'_reference_mcse']), summary['reference']['reference_mcse'][index])
            with (root/'extended/scs_100d_chains.csv').open(newline='') as stream:
                chains = list(csv.DictReader(stream))
            for row in chains:
                estimate = summary['cases'][row['method']]['chain_means'][int(row['chain'])]
                self.assertEqual(float(row['mean_x1_estimate']), estimate[2])
                self.assertEqual(float(row['x1_tail_probability_estimate']), estimate[3])
            with (root/'extended/paired_mse_differences.csv').open(newline='') as stream:
                paired = list(csv.DictReader(stream))
            self.assertEqual({row['metric'] for row in paired}, set(summary['functionals']))
            with (root/'extended/reference_values.csv').open(newline='') as stream:
                references = list(csv.DictReader(stream))
            self.assertEqual([row['exact'] for row in references], ['True', 'True', 'False', 'False'])
            page = (root/'extended/paper_values.html').read_text()
            self.assertIn('Reference values and precision', page)
            self.assertIn('P(X₁ &gt; 3.00)', page)
            self.assertIn('same four quantities', page)
        self.assertEqual(summary, before)

    def test_coordinate_references_are_estimated_and_precision_is_visible(self):
        summary = coordinate_summary(optimized=True)
        summary['reference']['reference_mcse'][2] = .1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = render_report(root, summary)
            self.assertIn('coordinate_mean', report['figures'])
            page = (root/'paper_values.html').read_text()
            self.assertIn('Reference precision matters.', page)
            self.assertIn('P(X₁ &gt; 3.00)', page)
            self.assertNotIn('DCS-RWM', page)
            with (root/'reference_values.csv').open(newline='') as stream:
                self.assertEqual([row['exact'] for row in csv.DictReader(stream)], ['True', 'True', 'False', 'False'])

    def test_original_target_choices_and_v2_replots_without_sampling(self):
        self.assertEqual(launcher.COMPARISON_TARGETS, ('skew-t-100d', 'skew-t-500d'))
        page = (launcher.ROOT/'launcher/index.html').read_text()
        self.assertIn('value="skew-t-100d"', page)
        self.assertIn('value="skew-t-500d"', page)
        for optimized in (False, True):
            summary = coordinate_summary(optimized=optimized)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'summary.json').write_text(json.dumps(summary))
                with patch('scripts.scs_100d_figures.render_report') as render, \
                        patch.object(run_experiment, 'run') as sampling:
                    self.assertEqual(run_experiment.main(['--plot-only', str(root)]), 0)
                    render.assert_called_once()
                    sampling.assert_not_called()

    def test_missing_or_inconsistent_reference_is_rejected(self):
        summary = coordinate_summary()
        with tempfile.TemporaryDirectory() as directory:
            summary.pop('reference')
            with self.assertRaisesRegex(ValueError, 'require recorded reference'):
                render_report(directory, summary)
            summary = coordinate_summary()
            summary['reference']['truth'][2] += 1.
            with self.assertRaisesRegex(ValueError, 'must match'):
                render_report(directory, summary)


if __name__ == '__main__':
    unittest.main()
