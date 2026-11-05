"""Version-10 reports distinguish matched VI counts from historical fits."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiment
from scripts.scs_100d_figures import render_report
from tests.test_v9_comparison_report import author_vi_summary


def matched_vi_summary(backend='jax'):
    data = author_vi_summary(backend, 'on')
    config = data['config']
    config.update(protocol='timed-sampler-comparison-v10' if backend == 'jax' else 'timed-numba-comparison-v10',
        vi='on', vi_enabled=True, vi_batch_policy='fixed', optimizer_steps=20, optimizer_samples=128,
        scs_vi=dict(optimizer_steps=20, optimizer_samples=128),
        vi_budget_policy='equal_updates_and_fixed_reference_clouds_v1')
    for fit in data['fitting'].values():
        fit.update(steps=20, training_draws=128, total_training_draws=128,
            batch_policy='fixed', training_batch_policy='fixed', gradient_updates=20,
            unique_reference_draws=128, objective_point_evaluations=2560)
    return data


class MatchedVIReportTests(unittest.TestCase):
    def test_v10_exports_equal_fixed_cloud_resources_for_both_backends(self):
        for backend in ('jax', 'numba'):
            data = matched_vi_summary(backend)
            before = deepcopy(data)
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = render_report(root, data)
                self.assertEqual(len(result['figures']), 5)
                with (root/'setup_times.csv').open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual({row['map'] for row in rows}, {'ball', 'sphere'})
                for row in rows:
                    self.assertEqual(row['training_batch_policy'], 'fixed')
                    self.assertEqual(int(row['gradient_updates']), 20)
                    self.assertEqual(int(row['unique_reference_draws']), 128)
                    self.assertEqual(int(row['objective_point_evaluations']), 2560)
                page = (root/'paper_values.html').read_text(encoding='utf-8')
                self.assertIn('Each map family reuses its own fixed reference cloud', page)
                self.assertIn('same number of draws, optimizer updates and objective-point evaluations', page)
                self.assertIn('Equal budgets do not make the objectives or clipping rules identical', page)
                self.assertNotIn('DCS retains fresh-batch VI', page)
                self.assertIn('SCS uses the unmodified authors', page)
                self.assertEqual(data, before)

    def test_v10_does_not_relabel_untuned_or_unmatched_data(self):
        for change in (lambda d: d['config'].update(vi='off'),
                       lambda d: d['config'].update(vi_batch_policy='fresh'),
                       lambda d: d['config']['scs_vi'].update(optimizer_steps=19),
                       lambda d: d['fitting']['ball'].update(batch_policy='fresh'),
                       lambda d: d['fitting']['sphere'].update(unique_reference_draws=127),
                       lambda d: d['fitting']['ball'].update(objective_point_evaluations=2559)):
            data = matched_vi_summary()
            change(data)
            with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'matched fixed-cloud'):
                render_report(Path(directory), data)

    def test_v10_replot_routes_without_sampling(self):
        for backend in ('jax', 'numba'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'summary.json').write_text(json.dumps(matched_vi_summary(backend)))
                with patch('scripts.scs_100d_figures.render_report') as render, \
                        patch.object(run_experiment, 'run') as sampling:
                    self.assertEqual(run_experiment.main(['--plot-only', str(root)]), 0)
                    render.assert_called_once()
                    sampling.assert_not_called()


if __name__ == '__main__':
    unittest.main()
