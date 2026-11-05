"""Report shared fixed DCS moves without relabelling historical calibration."""
from contextlib import redirect_stderr
from copy import deepcopy
import csv
from io import StringIO
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiment
from scripts import reproduce_scs_step_control as control
from scripts import reproduce_scs_100d as runner
from scripts.scs_100d_figures import render_report
from tests.test_scs_step_control import HARDWARE, SOURCES, primary_fixture
from tests.test_v10_comparison_report import matched_vi_summary


def fixed_dcs_summary(backend='jax', rule='fixed', radius=.07):
    data = matched_vi_summary(backend)
    config = data['config']
    config.update(protocol='timed-sampler-comparison-v11' if backend == 'jax' else 'timed-numba-comparison-v11',
                  dcs_step_rule=rule, dcs_fixed_move=radius if rule == 'fixed' else None,
                  scs_step_rule='fixed')
    config['step_tuning_policies']['SCS'] = 'author_scs_fixed_01'
    d = config['target']['dimension']
    historical = {'DCS-RWM': 'indicator_dual_averaging_0234',
                  'DCS-BW': 'indicator_dual_averaging_0234',
                  'DCS-BW-reflect': 'ball_esjd_pilot_v1',
                  'DCS-Billiard': 'own_kernel_radial_pilot_v1'}
    for method in historical:
        policy = 'fixed_dcs_move_v1' if rule == 'fixed' else historical[method]
        config['step_tuning_policies'][method] = policy
        case = data['cases'][method]
        for record in case['records']:
            record.update(tuning_policy=policy,
                          target_acceptance=.234 if policy == 'indicator_dual_averaging_0234' else None,
                          adaptation_during_sampling=False, warmup_iterations=config['warmup'])
            if rule == 'fixed':
                for key in list(record):
                    if key.startswith('radial_tuning_') or key.startswith('tuning_') and key != 'tuning_policy':
                        record.pop(key)
                scale = radius/math.sqrt(d+2.) if method == 'DCS-RWM' else radius
                record.update(initial_scale=scale, final_scale=scale, frozen_scale=scale,
                    fixed_step=scale, fixed_move_radius=radius, step_size_adapted=False,
                    fixed_move_normalization='BW, reflected BW and billiard radius h; RWM coordinate sigma=h/sqrt(d+2)',
                    proposal_coordinate_variance=radius**2/(d+2.))
        case['final_scales'] = [r['final_scale'] for r in case['records']]
    return data


class FixedDCSReportTests(unittest.TestCase):
    def test_fixed_moves_and_acceptance_controls_export_actual_policies(self):
        for backend, rule in (('jax', 'fixed'), ('numba', 'fixed'), ('jax', 'acceptance')):
            data = fixed_dcs_summary(backend, rule)
            before = deepcopy(data)
            with self.subTest(backend=backend, rule=rule), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                render_report(root, data)
                with (root/'paper_values.csv').open(newline='', encoding='utf-8') as stream:
                    rows = {r['method']: r for r in csv.DictReader(stream)}
                page = (root/'paper_values.html').read_text(encoding='utf-8')
                self.assertIn('Each map family reuses its own fixed reference cloud', page)
                self.assertIn('fixed h = 0.1', rows['SCS']['production_step_policy'])
                for method in ('DCS-RWM', 'DCS-BW', 'DCS-BW-reflect', 'DCS-Billiard'):
                    row = rows[method]
                    dimension = data['config']['target']['dimension']
                    self.assertEqual(row['dcs_step_rule'], rule)
                    if rule == 'fixed':
                        self.assertIn('no step adaptation', row['production_step_policy'])
                        self.assertNotIn('0.234', row['production_step_policy'])
                        self.assertEqual(float(row['dcs_fixed_move']), .07)
                        self.assertEqual(row['target_acceptance'], '')
                        self.assertAlmostEqual(float(row['proposal_coordinate_variance']), .07**2/(dimension+2.))
                        self.assertAlmostEqual(float(row['fixed_step']), .07/math.sqrt(dimension+2.) if method == 'DCS-RWM' else .07)
                    else:
                        if method in ('DCS-RWM', 'DCS-BW'):
                            self.assertIn('Acceptance-target 0.234', row['production_step_policy'])
                            self.assertEqual(float(row['target_acceptance']), .234)
                        elif method == 'DCS-Billiard':
                            self.assertIn('Own-production-kernel radial', row['production_step_policy'])
                        else:
                            self.assertIn('Ball-movement pilot', row['production_step_policy'])
                        self.assertNotIn('dcs_fixed_move', row)
                if rule == 'fixed':
                    self.assertIn('sigma=h/sqrt(d+2)', page)
                    self.assertIn('not accepted movement or mixing', page)
                    self.assertNotIn('DCS proposal scales are pretuned', page)
                    self.assertNotIn('DCS-RWM and ordinary BW target acceptance 0.234', page)
                    self.assertNotIn('Billiard pilots score', page)
                    self.assertNotIn('ball-movement pilot search', page)
                    self.assertNotIn('Own-kernel radial pilots and frozen steps', page)
                    self.assertFalse((root/'radial_tuning.csv').exists())
                    self.assertFalse((root/'reflected_tuning.csv').exists())
                self.assertEqual(data, before)

    def test_inconsistent_fixed_records_are_not_relabelled(self):
        changes = (
            lambda d: d['config']['step_tuning_policies'].update({'DCS-BW': 'indicator_dual_averaging_0234'}),
            lambda d: d['cases']['DCS-RWM']['records'][0].update(target_acceptance=.234),
            lambda d: d['cases']['DCS-BW']['records'][0].update(step_size_adapted=True),
            lambda d: d['cases']['DCS-RWM']['records'][0].update(final_scale=.07),
            lambda d: d['cases']['DCS-BW']['records'][0].pop('fixed_move_radius'),
            lambda d: d['cases']['DCS-BW'].update(final_scales=[.08]*3),
            lambda d: d['config']['step_tuning_policies'].update({'DCS-Billiard': 'own_kernel_radial_pilot_v1'}),
            lambda d: d['cases']['DCS-BW-reflect']['records'][0].update(step_size_adapted=True),
            lambda d: d['cases']['DCS-Billiard']['records'][0].update(radial_tuning_history=[{'scale': .07}]),
        )
        for change in changes:
            data = fixed_dcs_summary()
            change(data)
            with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'Version-11'):
                render_report(Path(directory), data)

    def test_v10_reports_keep_acceptance_policy_without_new_fixed_fields(self):
        data = matched_vi_summary()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render_report(root, data)
            with (root/'paper_values.csv').open(newline='', encoding='utf-8') as stream:
                rows = {r['method']: r for r in csv.DictReader(stream)}
            self.assertIn('0.234', rows['DCS-BW']['production_step_policy'])
            self.assertNotIn('dcs_fixed_move', rows['DCS-BW'])
            self.assertNotIn('Prespecified fixed DCS move', (root/'paper_values.html').read_text())

    def test_v11_scs_only_acceptance_control_does_not_describe_dcs_adaptation(self):
        data = fixed_dcs_summary()
        data['config'].update(algorithms=['SCS'], scs_step_rule='acceptance',
            step_tuning_policies={'SCS': 'indicator_dual_averaging_0234'},
            calibration_control_provenance={'maps_reused_without_refitting': True})
        data['cases'] = {'SCS': data['cases']['SCS']}
        data['method_maps'] = {'SCS': data['method_maps']['SCS']}
        with tempfile.TemporaryDirectory() as directory:
            render_report(Path(directory), data)
            page = (Path(directory)/'paper_values.html').read_text()
            self.assertIn('This report contains only SCS', page)
            self.assertIn('SCS uses acceptance-target 0.234 calibration', page)
            self.assertNotIn('SCS, DCS-RWM and ordinary DCS-BW use acceptance', page)
            self.assertNotIn('billiard tunes its own radial moves', page)

    def test_v11_plot_only_routes_without_sampling(self):
        for backend in ('jax', 'numba'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'summary.json').write_text(json.dumps(fixed_dcs_summary(backend)))
                with patch('scripts.scs_100d_figures.render_report') as render, patch.object(run_experiment, 'run') as run:
                    self.assertEqual(run_experiment.main(['--plot-only', str(root)]), 0)
                    render.assert_called_once()
                    run.assert_not_called()

    def test_dcs_options_are_forwarded_only_for_current_comparisons(self):
        for experiment in ('sampler-comparison', 'sampler-comparison-numba'):
            with patch('scripts.reproduce_comparison.main', return_value=71) as run:
                self.assertEqual(run_experiment.main([experiment, '--dcs-step-rule', 'fixed', '--dcs-fixed-move', '.09', '--dry-run']), 71)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index('--dcs-step-rule')+1], 'fixed')
                self.assertEqual(float(command[command.index('--dcs-fixed-move')+1]), .09)
        for experiment in ('skew-t-100d-fixed', 'eight-schools'):
            for option in (['--dcs-step-rule', 'acceptance'], ['--dcs-fixed-move', '.07']):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    run_experiment.main([experiment, *option, '--dry-run'])

    def test_v11_scs_control_keeps_original_fitting_and_dcs_configuration(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, 'cpu_info', return_value=HARDWARE), \
                patch.object(runner, 'source_hashes', return_value=SOURCES.copy()), \
                patch.object(control, 'verify_vendor', return_value=True):
            root = Path(directory)/'primary'
            files = primary_fixture(root)
            config = files['configuration.json']
            config.update(protocol='timed-sampler-comparison-v11', dcs_step_rule='fixed', dcs_fixed_move=.07)
            files['manifest.json']['protocol'] = config['protocol']
            # The fixture shares no object identity after serialization; keep all recorded configs equal.
            files['manifest.json']['config'] = deepcopy(config)
            files['summary.json']['config'] = deepcopy(config)
            for name, value in files.items():
                (root/name).write_text(json.dumps(value))
            result, maps, fitting = control.prepare_control(root)
            self.assertEqual(result['protocol'], 'timed-sampler-comparison-v11')
            self.assertEqual(result['dcs_fixed_move'], .07)
            self.assertEqual(result['algorithms'], ['SCS'])
            self.assertEqual(result['step_tuning_policies'], {'SCS': 'author_scs_fixed_01'})
            self.assertEqual(maps, files['maps.json']['maps'])
            self.assertTrue(all(f['map_reused_without_refitting'] for f in fitting.values()))


if __name__ == '__main__':
    unittest.main()
