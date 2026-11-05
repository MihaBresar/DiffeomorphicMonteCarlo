"""Keep current own-kernel pilots distinct from historical reflected pilots."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiment
from scripts import reproduce_comparison as suite
from scripts.scs_100d_figures import render_report
from tests.test_radial_ess_comparison import radial_summary


def own_kernel_summary(backend='jax', vi='on'):
    summary = radial_summary(backend, vi)
    summary['config']['protocol'] = ('timed-sampler-comparison-v8' if backend == 'jax'
                                     else 'timed-numba-comparison-v8')
    count, remainder = divmod(summary['config']['warmup'], 8)
    for method in ('DCS-Billiard', 'SCS'):
        for record in summary['cases'][method]['records']:
            record.pop('tuning_history', None)
            record.update(calibration_algorithm=method,
                radial_tuning_pilot_iterations=summary['config']['warmup'],
                radial_tuning_rounds=2, radial_tuning_windows=8,
                radial_tuning_selected_score=.012345678912345,
                radial_tuning_search_status='interior_refinement_candidate_won',
                radial_tuning_history=[dict(stage='coarse' if trial < 5 else 'refinement',
                    round=1 if trial < 5 else 2, candidate=trial if trial < 5 else trial-5,
                    trial_order=trial+1, scale=.1*(trial+1),
                    iterations=count+(trial < remainder),
                    radial_score=.012345678912345*(trial+1), acceptance=.7,
                    selected=trial in (2,6)) for trial in range(8)])
    return summary


class OwnKernelCalibrationReportTests(unittest.TestCase):
    def test_current_policy_has_equal_budget_and_preserves_all_map_defaults(self):
        for backend in ('jax','numba'):
            for vi in ('on',):
                config = suite.configuration('paper', comparison=backend, vi=vi)
                self.assertEqual(config['step_tuning_policies']['DCS-Billiard'], 'own_kernel_radial_pilot_v1')
                self.assertEqual(config['step_tuning_policies']['SCS'], 'author_scs_fixed_01')
                self.assertEqual(config['warmup'],1500)
                self.assertEqual(config['initial_scale'],.1)
                self.assertEqual(config['beta'],2.)
                self.assertEqual(config['latitude'],1.1)
                self.assertEqual(config['ball_map_policy'],'mobius' if vi=='on' else 'vanilla')
        with self.assertRaisesRegex(ValueError,'at least eight'):
            suite.configuration('quick',overrides={'warmup':7})

    def test_reports_export_actual_kernel_radial_scores_without_reflected_mislabel(self):
        for backend,vi in (('jax','on'),('numba','off')):
            summary = own_kernel_summary(backend,vi)
            before=deepcopy(summary)
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                report=render_report(root,summary)
                def rows(name):
                    with (root/name).open(newline='',encoding='utf-8') as source:
                        return list(csv.DictReader(source))
                pilots=rows('radial_tuning.csv')
                self.assertEqual({r['method'] for r in pilots},{'SCS','DCS-Billiard'})
                self.assertTrue(all(r['pilot_kernel']==r['method'] for r in pilots))
                for method in ('SCS','DCS-Billiard'):
                    for record in summary['cases'][method]['records']:
                        selected=[r for r in pilots if r['method']==method and int(r['chain'])==record['chain']]
                        self.assertEqual(len(selected),8)
                        self.assertEqual(sum(int(r['iterations']) for r in selected),summary['config']['warmup'])
                        self.assertEqual([float(r['radial_score']) for r in selected],
                                         [r['radial_score'] for r in record['radial_tuning_history']])
                self.assertEqual({r['method'] for r in rows('reflected_tuning.csv')},{'DCS-BW-reflect'})
                self.assertIn('functional_ess.csv',report['csv_files'])
                self.assertNotIn('angular_tuning.csv',report['csv_files'])
                values=rows('paper_values.csv')
                for row in values:
                    if row['method'] in ('SCS','DCS-Billiard'):
                        self.assertIn('Own-production-kernel',row['production_step_policy'])
                        self.assertNotIn('0.234',row['production_step_policy'])
                        self.assertNotIn('Reflected-BW',row['production_step_policy'])
                page=(root/'paper_values.html').read_text()
                self.assertIn('Own-kernel radial pilots and frozen steps',page)
                self.assertIn('equal pilot transition counts, not necessarily equal pilot seconds',page)
                self.assertNotIn('The pilot table therefore scores reflected-BW movement',page)
                self.assertEqual(summary,before)

    def test_missing_or_wrong_kernel_history_cannot_be_relabelled_current(self):
        data=own_kernel_summary()
        data['cases']['DCS-Billiard']['records'][0]['calibration_algorithm']='DCS-BW-reflect'
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError,'own-kernel radial pilot provenance'):
            render_report(Path(directory),data)

    def test_v8_replot_routes_without_sampling(self):
        for backend in ('jax','numba'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                (root/'summary.json').write_text(json.dumps(own_kernel_summary(backend)))
                with patch('scripts.scs_100d_figures.render_report') as render, patch.object(run_experiment,'run') as run:
                    self.assertEqual(run_experiment.main(['--plot-only',str(root)]),0)
                    render.assert_called_once();run.assert_not_called()


if __name__=='__main__':unittest.main()
