"""Route the added boundary comparison without changing the default experiment."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import START_HERE as launcher
import run_experiment
from dcs_bench.cpu import cpu_info


EXPERIMENT = 'skew-t-100d-boundary'


class BoundaryRoutingTests(unittest.TestCase):
    def test_launcher_uses_separate_fresh_runner_with_matching_budgets(self):
        workers = min(2, cpu_info()['capacity'])
        for preset in ('quick', 'reproduce', 'paper'):
            command, = launcher.commands_for_run(preset, EXPERIMENT, Path('results/test'),
                workers=workers, ball_map_policy='vanilla')
            with self.subTest(preset=preset):
                self.assertEqual(command[2], str(launcher.ROOT/'scripts/reproduce_boundary_100d.py'))
                self.assertEqual(command[command.index('--preset')+1], preset)
                self.assertEqual(command[command.index('--workers')+1], str(workers))
                self.assertEqual(command[command.index('--output-dir')+1], str(Path('results/test')/EXPERIMENT))
                self.assertNotIn('--ball-map-policy', command)
        for options in ({'resume': True}, {'plot_only': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                launcher.commands_for_run('quick', EXPERIMENT, Path('results/test'), **options)

    def test_cli_dispatches_boundary_overrides_to_dedicated_runner(self):
        module = types.ModuleType('scripts.reproduce_boundary_100d')
        module.main = Mock(return_value=81)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'new run'
            with patch.dict(sys.modules, {'scripts.reproduce_boundary_100d': module}), \
                    patch.object(run_experiment, 'run') as generic:
                status = run_experiment.main([EXPERIMENT, '--preset', 'paper', '--workers', '1',
                    '--runs', '4', '--time-budget', '1.25', '--warmup', '1500', '--thin', '10',
                    '--optimizer-steps', '17', '--optimizer-samples', '42', '--seed', '91',
                    '--output-dir', str(output), '--dry-run'])
            self.assertEqual(status, 81)
            generic.assert_not_called()
            self.assertEqual(module.main.call_args.args[0], [
                '--preset', 'paper', '--workers', '1', '--output-dir', str(output), '--dry-run',
                '--runs', '4', '--time-budget', '1.25', '--warmup', '1500', '--thin', '10',
                '--optimizer-steps', '17', '--optimizer-samples', '42', '--seed', '91'])

    def test_boundary_cli_rejects_incompatible_options(self):
        module = types.ModuleType('scripts.reproduce_boundary_100d')
        module.main = Mock()
        for options in (['--resume'], ['--ball-map-policy', 'vanilla'], ['--iterations', '1000'],
                        ['--nuts-samples', '500'], ['--reference-samples', '500']):
            with self.subTest(options=options), \
                    patch.dict(sys.modules, {'scripts.reproduce_boundary_100d': module}), \
                    redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    run_experiment.main([EXPERIMENT]+options)
                self.assertEqual(caught.exception.code, 2)
        module.main.assert_not_called()

    def test_boundary_saved_summary_uses_shared_renderer_without_resampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = {'config': {'protocol': 'timed-boundary-dcs-scs-v1'}}
            (output/'summary.json').write_text(json.dumps(summary))
            with patch('scripts.scs_100d_figures.render_report') as render, \
                    patch.object(run_experiment, 'run') as sample, redirect_stdout(StringIO()):
                self.assertEqual(run_experiment.main(['--plot-only', str(output)]), 0)
            render.assert_called_once_with(output, summary)
            sample.assert_not_called()

    def test_cli_lists_new_experiment_once(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(run_experiment.main(['--list']), 0)
        self.assertEqual(output.getvalue().splitlines().count(EXPERIMENT), 1)
        self.assertIn('skew-t-100d-fixed', output.getvalue().splitlines())


if __name__ == '__main__':
    unittest.main()
