"""Report actual author-VI and method-specific calibration without relabelling history."""
from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiment
from scripts.scs_100d_figures import render_report
from tests.test_radial_ess_comparison import radial_summary
from tests.test_scs_100d_figures import example_summary
from tests.test_boundary_v3_reports import boundary_summary


def author_vi_summary(backend='jax', vi='on'):
    data = radial_summary(backend, vi)
    config = data['config']
    config.update(protocol='timed-sampler-comparison-v9' if backend == 'jax' else
                  'timed-numba-comparison-v9', fit_seed=42,
                  step_tuning_policies={'SCS': 'indicator_dual_averaging_0234',
                      'DCS-Billiard': 'own_kernel_radial_pilot_v1',
                      'DCS-BW-reflect': 'ball_esjd_pilot_v1'})
    data['fitting']['sphere'].update(seed=0, learning_rate=.01, map_policy='SCS-paper-VI',
        batch_policy='fixed', training_batch_policy='fixed', vi_coordinate_clip=1000.,
        gradient_clipping=False, source_commit='author-revision', full_author_vi_budget=False)
    for record in data['cases']['DCS-Billiard']['records']:
        record.pop('tuning_history', None)
        record.update(calibration_algorithm='DCS-Billiard',
            radial_tuning_pilot_iterations=config['warmup'],
            radial_tuning_history=[dict(stage='coarse', round=1, candidate=0,
                trial_order=1, scale=.4, iterations=config['warmup'],
                radial_score=.2, acceptance=.3, selected=True)])
    return data


class AuthorViReportTests(unittest.TestCase):
    def test_legacy_entry_points_report_recorded_author_fitting_without_changing_step_rules(self):
        matched = example_summary()
        matched['config'].update(protocol='timed-matched-dcs-scs-v3', mode='time', time_budget=1.,
                                 implementation_revision='jax-dcs-exact-ball-v1')
        for case in matched['cases'].values():
            for row in case['records']:
                row.update(discarded_iterations=334, estimation_iterations=666,
                           retained_draws=67, warmup_iterations=100, target_acceptance=.234)
                row.pop('checkpoints', None)
        for historical in (matched, boundary_summary()):
            data = deepcopy(historical)
            data['config'].update(sphere_map_policy='author-paper-vi', scs_fitting_revision='author-vi-v1',
                                  scs_vi={'optimizer_steps': 1000, 'optimizer_samples': 2000})
            data['fitting']['sphere'].update(seed=0, map_policy='SCS-paper-VI',
                vi_coordinate_clip=1000., gradient_clipping=False, source_commit='author-revision')
            before = deepcopy(data)
            with self.subTest(protocol=data['config']['protocol']), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                render_report(root, data)
                page = (root/'paper_values.html').read_text(encoding='utf-8')
                self.assertIn('SCS uses the unmodified authors', page)
                self.assertIn('one fixed training batch, seed 0', page)
                self.assertIn('independently recorded budgets', page)
                self.assertNotIn('same fresh-batch reverse-KL fitting framework', page)
                self.assertNotIn('Map fitting, recording and time budgets are matched', page)
                self.assertNotIn('matched map fitting', page)
                self.assertIn('0.234', page)
                with (root/'setup_times.csv').open(newline='', encoding='utf-8') as stream:
                    setup = {row['map']: row for row in csv.DictReader(stream)}
                self.assertEqual(setup['sphere']['seed'], '0')
                self.assertEqual(setup['sphere']['source_commit'], 'author-revision')
                self.assertEqual(setup['sphere']['training_batch_policy'], 'fixed')
                self.assertEqual(int(setup['sphere']['total_training_draws']), 2000)
                self.assertEqual(int(setup['ball']['total_training_draws']), 2002000)
                self.assertEqual(data, before)
        with tempfile.TemporaryDirectory() as directory:
            render_report(Path(directory), matched)
            page = (Path(directory)/'paper_values.html').read_text(encoding='utf-8')
            self.assertIn('same fresh-batch reverse-KL fitting framework', page)
            self.assertNotIn('SCS uses the unmodified authors', page)

    def test_scs_acceptance_and_billiard_radial_pilots_are_reported_separately(self):
        for backend, vi in (('jax', 'on'), ('numba', 'off')):
            data = author_vi_summary(backend, vi)
            before = deepcopy(data)
            with self.subTest(backend=backend, vi=vi), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = render_report(root, data)
                self.assertEqual(len(result['figures']), 5)
                def rows(name):
                    with (root/name).open(newline='', encoding='utf-8') as stream:
                        return list(csv.DictReader(stream))
                methods = {row['method']: row for row in rows('paper_values.csv')}
                self.assertIn('Acceptance-target 0.234', methods['SCS']['production_step_policy'])
                self.assertNotIn('radial', methods['SCS']['production_step_policy'])
                self.assertIn('Own-production-kernel radial', methods['DCS-Billiard']['production_step_policy'])
                self.assertEqual({row['method'] for row in rows('radial_tuning.csv')}, {'DCS-Billiard'})
                self.assertEqual({row['method'] for row in rows('reflected_tuning.csv')}, {'DCS-BW-reflect'})
                setup = {row['map']: row for row in rows('setup_times.csv')}
                self.assertEqual(setup['sphere']['map_policy'], 'SCS-paper-VI')
                self.assertEqual(setup['sphere']['training_batch_policy'], 'fixed')
                self.assertEqual(float(setup['sphere']['vi_coordinate_clip']), 1000.)
                self.assertEqual(setup['sphere']['seed'], '0')
                page = (root/'paper_values.html').read_text()
                self.assertNotIn('equal optimizer and fresh-draw budgets', page)
                self.assertNotIn('Billiard and SCS tune their own kernels', page)
                self.assertNotIn('Billiard and SCS each tune', page)
                if vi == 'on':
                    self.assertIn('SCS uses the unmodified authors', page)
                    self.assertIn('distinct fitting protocols', page)
                else:
                    self.assertIn('VI is disabled for both map families', page)
                self.assertEqual(data, before)

    def test_missing_billiard_pilot_is_rejected_but_scs_needs_no_radial_pilot(self):
        data = author_vi_summary()
        del data['cases']['DCS-Billiard']['records'][0]['radial_tuning_history']
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'pilot provenance'):
            render_report(Path(directory), data)

    def test_fixed_author_step_control_has_one_sampler_and_no_tuning_claim(self):
        data = author_vi_summary()
        data['config'].update(algorithms=['SCS'],
            step_tuning_policies={'SCS': 'author_scs_fixed_01'},
            calibration_control_provenance={'maps_reused_without_refitting': True})
        data['cases'] = {'SCS': data['cases']['SCS']}
        data['method_maps'] = {'SCS': data['method_maps']['SCS']}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render_report(root, data)
            page = (root/'paper_values.html').read_text()
            self.assertIn('one sampler, JAX', page)
            self.assertIn('skew-t example step h=0.1', page)
            self.assertIn('step-size sensitivity control uses', page)
            self.assertNotIn('SCS acceptance calibration follows', page)
            self.assertFalse((root/'radial_tuning.csv').exists())
            with (root/'paper_values.csv').open(newline='', encoding='utf-8') as stream:
                row, = list(csv.DictReader(stream))
            self.assertIn('fixed h = 0.1', row['production_step_policy'])
            self.assertIn('no step adaptation', row['production_step_policy'])

    def test_fixed_default_five_method_primary_does_not_claim_reused_maps(self):
        data = author_vi_summary()
        data['config']['step_tuning_policies']['SCS'] = 'author_scs_fixed_01'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render_report(root, data)
            page = (root/'paper_values.html').read_text()
            self.assertIn('five samplers, JAX', page)
            self.assertIn('skew-t example step h=0.1', page)
            self.assertIn('with no step adaptation', page)
            self.assertNotIn('step-size sensitivity control uses', page)
            self.assertNotIn('exact recorded SCS map as the primary', page)
            self.assertNotIn('SCS acceptance calibration follows', page)
            with (root/'paper_values.csv').open(newline='', encoding='utf-8') as stream:
                rows = {row['method']: row for row in csv.DictReader(stream)}
            self.assertIn('fixed h = 0.1', rows['SCS']['production_step_policy'])
            self.assertIn('Own-production-kernel radial', rows['DCS-Billiard']['production_step_policy'])

    def test_scs_fixed_alias_is_forwarded_only_for_earlier_fitted_comparisons(self):
        for experiment in ('sampler-comparison', 'sampler-comparison-numba'):
            with self.subTest(experiment=experiment), \
                    patch('scripts.reproduce_comparison.main', return_value=71) as comparison:
                self.assertEqual(run_experiment.main([experiment, '--scs-step-rule', 'fixed', '--dry-run']), 71)
                command = comparison.call_args.args[0]
                self.assertEqual(command[command.index('--scs-step-rule')+1], 'fixed')
            with self.subTest(experiment=experiment, rejected='acceptance'), \
                    patch('scripts.reproduce_comparison.main') as comparison, \
                    redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                run_experiment.main([experiment, '--scs-step-rule', 'acceptance', '--dry-run'])
            self.assertEqual(error.exception.code, 2)
            comparison.assert_not_called()
        for experiment in ('reported-zero-centre', 'reported-zero-centre-numba',
                           'skew-t-100d-fixed', 'eight-schools'):
            with self.subTest(experiment=experiment), \
                    patch('scripts.reproduce_reported_study.main') as reported, \
                    patch('scripts.reproduce_comparison.main') as comparison, \
                    redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                run_experiment.main([experiment, '--scs-step-rule', 'fixed', '--dry-run'])
            self.assertEqual(error.exception.code, 2)
            reported.assert_not_called()
            comparison.assert_not_called()

    def test_v9_replot_routes_without_sampling_for_both_backends(self):
        for backend in ('jax', 'numba'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'summary.json').write_text(json.dumps(author_vi_summary(backend)))
                with patch('scripts.scs_100d_figures.render_report') as render, \
                        patch.object(run_experiment, 'run') as sampling:
                    self.assertEqual(run_experiment.main(['--plot-only', str(root)]), 0)
                    render.assert_called_once()
                    sampling.assert_not_called()


if __name__ == '__main__':
    unittest.main()
