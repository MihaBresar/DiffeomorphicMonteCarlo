"""Show the mixture's two pilot phases and keep historical report meanings."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_experiment
from scripts.scs_100d_figures import COLORS, render_report
from tests.test_asymmetry_reports import coordinate_summary


def rotation_summary(optimized=False):
    summary = coordinate_summary(optimized=optimized, all_coordinates=True)
    summary['config'].update(protocol='timed-numba-comparison-v4' if optimized else 'timed-sampler-comparison-v4',
                             mixture_probability=.5)
    name = 'DCS-BW-rotate'
    summary['config']['algorithms'].append(name)
    case = deepcopy(summary['cases']['DCS-BW-reflect'])
    for record in case['records']:
        # A total of 100 pilot transitions is divided, not duplicated.
        record['tuning_history'] = [dict(round=1, candidate=1, scale=.1,
            iterations=50, ball_esjd=.012345678912345, acceptance=.8, selected=True)]
        record.update(final_scale=.1, frozen_scale=.1, angular_scale=.8,
            angular_angle=2*np.arctan(.8), mixture_probability=.5,
            bw_warmup_iterations=50, angular_warmup_iterations=50,
            angular_tuning_selected_esjd=1.2345678912345,
            angular_tuning_search_status='current_candidate_won_final_round',
            angular_tuning_history=[dict(round=1, candidate=1, scale=.8,
                iterations=50, angular_esjd=1.2345678912345, acceptance=.7, selected=True)])
    case['final_scales'] = [.1]*case['chains']
    summary['cases'][name] = case
    return summary


class RotationReportTests(unittest.TestCase):
    def test_mixture_has_all_figures_exact_metrics_and_separate_tuning_exports(self):
        for optimized in (False, True):
            summary = rotation_summary(optimized)
            before = deepcopy(summary)
            with self.subTest(optimized=optimized), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                report = render_report(root, summary)
                self.assertEqual(len(report['figures']), 10)
                self.assertIn('angular_tuning.csv', report['csv_files'])
                self.assertIn('reflected_tuning.csv', report['csv_files'])
                with (root/'paper_values.csv').open(newline='', encoding='utf-8') as stream:
                    cases = list(csv.DictReader(stream))
                self.assertEqual([row['method'] for row in cases], summary['config']['algorithms'])
                mixture, = [row for row in cases if row['method'] == 'DCS-BW-rotate']
                self.assertNotIn('0.234', mixture['production_step_policy'])
                self.assertIn('50/50', mixture['production_step_policy'])
                self.assertEqual(mixture['execution_backend'], 'Numba' if optimized else 'JAX')
                self.assertEqual(float(mixture['mixture_probability']), .5)
                for row in cases:
                    for index, metric in enumerate(summary['functionals']):
                        self.assertEqual(float(row[metric+'_mse']), summary['cases'][row['method']]['mse'][index])
                with (root/'angular_tuning.csv').open(newline='', encoding='utf-8') as stream:
                    pilots = list(csv.DictReader(stream))
                self.assertEqual(len(pilots), 3)
                self.assertEqual({row['method'] for row in pilots}, {'DCS-BW-rotate'})
                self.assertEqual(float(pilots[0]['angular_esjd']), 1.2345678912345)
                with (root/'reflected_tuning.csv').open(newline='', encoding='utf-8') as stream:
                    bw_pilots = list(csv.DictReader(stream))
                self.assertEqual({row['method'] for row in bw_pilots}, {'DCS-BW-reflect', 'DCS-BW-rotate'})
                with (root/'scs_100d_chains.csv').open(newline='', encoding='utf-8') as stream:
                    chains = list(csv.DictReader(stream))
                for row in chains:
                    self.assertNotIn('angular_tuning_history', row)
                    self.assertNotIn('tuning_history', row)
                    if row['method'] == 'DCS-BW-rotate':
                        self.assertEqual(int(row['bw_warmup_iterations'])+int(row['angular_warmup_iterations']), 100)
                        self.assertEqual(float(row['angular_angle']), 2*np.arctan(.8))
                page = (root/'paper_values.html').read_text(encoding='utf-8')
                for text in ('Rotation pilot tuning and frozen angles', 'frozen angle degrees',
                             'physical distance from the fitted centre',
                             'Only the chosen transition is performed and counted', 'same eight quantities'):
                    self.assertIn(text, page)
                if not optimized:
                    self.assertIn('five samplers, JAX', page)
            self.assertEqual(summary, before)
        self.assertEqual(len(COLORS.values()), len(set(COLORS.values())))

    def test_v4_replots_do_not_start_samplers(self):
        for optimized in (False, True):
            with self.subTest(optimized=optimized), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'summary.json').write_text(json.dumps(rotation_summary(optimized)), encoding='utf-8')
                with patch('scripts.scs_100d_figures.render_report') as render, \
                        patch.object(run_experiment, 'run') as sampling:
                    self.assertEqual(run_experiment.main(['--plot-only', str(root)]), 0)
                    render.assert_called_once()
                    sampling.assert_not_called()

    def test_launcher_explains_both_implementations_and_frozen_split_tuning(self):
        page = (Path(__file__).resolve().parents[1]/'launcher/index.html').read_text(encoding='utf-8')
        for text in ('Reported study: zero-centre DCS vs original SCS (JAX)',
                     'Reported-study parameters: Numba extension',
                     'Earlier fitted-map comparison: five samplers (JAX)',
                     'Earlier fitted-map comparison: Numba DCS vs SCS',
                     'functional ESS', 'All sampling steps are frozen'):
            self.assertIn(text, page)
        self.assertNotIn('DCS-BW-rotate', page)
        self.assertNotIn('DCS-RWM-rotate', page)


if __name__ == '__main__':
    unittest.main()
