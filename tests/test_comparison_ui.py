"""Route both comparison modes and report target/VI/backend choices honestly."""
from contextlib import redirect_stderr
from copy import deepcopy
import csv
from io import StringIO
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import START_HERE as launcher
import run_experiment
from scripts.scs_100d_figures import render_report
from tests.test_boundary_v3_reports import boundary_summary


def comparison_summary(optimized=False, vi='on'):
    summary = boundary_summary()
    if not optimized:
        summary['cases']['DCS-RWM'] = deepcopy(summary['cases']['DCS-BW'])
    dimension = 100 if optimized else 500
    target = {'kind': 'skew_t', 'dimension': dimension, 'nu': 3., 'alpha': {'0': 100., '1': -100.}}
    names = ['DCS-BW', 'DCS-BW-reflect', 'SCS'] if optimized else ['DCS-RWM', 'DCS-BW', 'SCS', 'DCS-BW-reflect']
    summary['config'].update(protocol='timed-numba-comparison-v1' if optimized else 'timed-sampler-comparison-v1',
        comparison='numba' if optimized else 'jax', target=target,
        target_title=f'{dimension}D skewed Student-t, ν = 3',
        vi=vi, beta=2., threshold=100.123456789, algorithms=names)
    summary['maps']['ball']['beta'] = summary['config']['beta']
    if vi == 'off':
        for fitting in summary['fitting'].values():
            fitting.update(seconds=0., steps=0, samples_per_step=0, total_training_draws=0)
    return summary


class ComparisonUiTests(unittest.TestCase):
    def test_launcher_forwards_every_mode_target_and_vi_combination(self):
        for experiment in launcher.COMPARISONS:
            for target in launcher.COMPARISON_TARGETS:
                for vi in ('on', 'off'):
                    with self.subTest(experiment=experiment, target=target, vi=vi):
                        command, = launcher.commands_for_run('paper', experiment, Path('results/new'),
                            comparison_target=target, vi=vi, seed_batch=3)
                        self.assertEqual(command[2], str(launcher.ROOT/'scripts/reproduce_comparison.py'))
                        self.assertEqual(command[command.index('--target')+1], target)
                        self.assertEqual(command[command.index('--vi')+1], vi)
                        self.assertEqual(command[command.index('--comparison')+1],
                                         'numba' if experiment.endswith('-numba') else 'jax')
                        self.assertEqual(command[command.index('--seed-batch')+1], '3')
                        self.assertNotIn('--ball-map-policy', command)
        for extra in ({'comparison_target': 'unknown'}, {'vi': False}, {'vi': 'unknown'}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                launcher.commands_for_run('quick', 'sampler-comparison', Path('unused'), **extra)

    def test_controller_preserves_both_user_choices_in_saved_invocation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(launcher, 'RUNS', Path(directory)), \
                patch.object(launcher, 'environment_ready', return_value=True):
            controller = launcher.Controller()
            with patch.object(controller, 'start_job', side_effect=lambda work, status: work()), \
                    patch.object(controller, 'run_command') as execute:
                controller.run(experiment='sampler-comparison-numba', comparison_target='skew-t-500d', vi='off')
                saved = json.loads((Path(directory)/controller.current_run/'launcher-run.json').read_text(encoding='utf-8'))
                self.assertEqual(saved['comparison_target'], 'skew-t-500d')
                self.assertEqual(saved['vi'], 'off')
                command = execute.call_args.args[0]
                self.assertEqual(command[command.index('--target')+1], 'skew-t-500d')
                self.assertEqual(command[command.index('--vi')+1], 'off')
        command, = launcher.commands_for_run('paper', 'all', Path('unused'), vi='off')
        self.assertEqual(command[command.index('--vi')+1], 'off')

    def test_cli_dispatches_comparison_options_and_rejects_unused_controls(self):
        module = types.ModuleType('scripts.reproduce_comparison')
        module.main = Mock(return_value=71)
        for experiment in launcher.COMPARISONS:
            with self.subTest(experiment=experiment), patch.dict(sys.modules, {'scripts.reproduce_comparison': module}):
                self.assertEqual(run_experiment.main([experiment, '--target', 'skew-t-500d', '--vi', 'off',
                    '--seed-batch', '2', '--workers', '1', '--warmup', '1500', '--dry-run']), 71)
                command = module.main.call_args.args[0]
                self.assertEqual(command[command.index('--target')+1], 'skew-t-500d')
                self.assertEqual(command[command.index('--vi')+1], 'off')
                self.assertEqual(command[command.index('--warmup')+1], '1500')
                self.assertIn('--dry-run', command)
        for arguments in (['garch11', '--vi', 'off'], ['skew-t-100d-fixed', '--target', 'skew-t-500d'],
                          ['sampler-comparison', '--iterations', '100'],
                          ['sampler-comparison', '--beta', '1'],
                          ['sampler-comparison', '--ball-map-policy', 'vanilla']):
            with self.subTest(arguments=arguments), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                run_experiment.main(arguments)

    def test_new_protocol_replot_does_not_sample(self):
        for optimized in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                summary = comparison_summary(optimized)
                (root/'summary.json').write_text(json.dumps(summary))
                with patch('scripts.scs_100d_figures.render_report') as render, \
                        patch.object(run_experiment, 'run') as sampling:
                    self.assertEqual(run_experiment.main(['--plot-only', str(root)]), 0)
                    render.assert_called_once()
                    sampling.assert_not_called()

    def test_reports_keep_all_methods_metrics_and_backend_vi_distinctions(self):
        from matplotlib.figure import Figure
        original_savefig = Figure.savefig
        for optimized, vi in ((False, 'on'), (True, 'off'), (True, 'on')):
            summary = comparison_summary(optimized, vi)
            before = deepcopy(summary)
            saved_titles = {}
            def observe_figure(figure, filename, *args, **kwargs):
                saved_titles[Path(filename).stem] = figure._suptitle.get_text()
                return original_savefig(figure, filename, *args, **kwargs)
            with self.subTest(optimized=optimized, vi=vi), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(Figure, 'savefig', new=observe_figure):
                    report = render_report(root, summary)
                self.assertEqual(report['figures'], ['VsSCS', 'scs_100d_estimates', 'scs_100d_runtime', 'scs_100d_bias_variance'])
                self.assertEqual(set(saved_titles), set(report['figures']))
                for title in saved_titles.values():
                    self.assertIn(summary['config']['target_title'], title)
                    self.assertNotIn('500D' if optimized else '100D', title)
                with (root/'paper_values.csv').open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([row['method'] for row in rows], summary['config']['algorithms'])
                for row in rows:
                    method = row['method']
                    self.assertEqual(float(row['mean_norm_mse']), summary['cases'][method]['mse'][1])
                    self.assertEqual(int(row['total_iterations']), 3600)
                    self.assertEqual(row['vi'], vi)
                    self.assertEqual(row['execution_backend'], 'Numba' if optimized and method != 'SCS' else 'JAX')
                    if method == 'DCS-BW-reflect':
                        self.assertNotIn('0.234', row['production_step_policy'])
                    else:
                        self.assertIn('0.234', row['production_step_policy'])
                self.assertIn('reflected_tuning.csv', report['csv_files'])
                page = (root/'paper_values.html').read_text(encoding='utf-8')
                self.assertIn(summary['config']['target_title'], page)
                self.assertNotIn('Numerical step-size guard reached.', page)
                if optimized:
                    self.assertIn('VI is '+('enabled' if vi == 'on' else 'disabled')+' for both map families', page)
                    self.assertNotIn('DCS-RWM', page)
                    self.assertNotIn('fit only the scale R', page)
                    self.assertIn('no full physical-state recording', page)
                    self.assertNotIn('All three samplers use compiled JAX CPU scans', page)
                    self.assertNotIn('500D skewed Student-t', page)
                else:
                    self.assertIn('Common-framework comparison', page)
                    self.assertIn('500D skewed Student-t', page)
                    self.assertIn('DCS-RWM', page)
            self.assertEqual(summary, before)

    def test_numerical_guard_warning_includes_scs_without_changing_statistics(self):
        summary = comparison_summary(optimized=True, vi='on')
        row = summary['cases']['SCS']['records'][0]
        row.update(scale_ceiling_reached=True, step_scale_ceiling=math.exp(15.),
                   final_scale=math.exp(15.), frozen_scale=math.exp(15.), scale=math.exp(15.))
        before = deepcopy(summary)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render_report(root, summary)
            page = (root/'paper_values.html').read_text(encoding='utf-8')
            self.assertIn('Numerical step-size guard reached.', page)
            self.assertIn('SCS: 1/3 chains', page)
            self.assertIn('calibration saturation, not evidence of an optimal step', page)
        self.assertEqual(summary, before)


if __name__ == '__main__':
    unittest.main()
