"""Check GUI dispatch preserves the scientific pipeline and saved-run settings."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import START_HERE as launcher
from dcs_bench.cpu import cpu_info


class LauncherTests(unittest.TestCase):
    def test_tuning_audit_runs_separate_fresh_pipeline(self):
        command, = launcher.commands_for_run('quick', 'fixed-step-audit', Path('results/test'))
        self.assertIn(str(launcher.ROOT/'scripts/audit_fixed_steps.py'), command)
        self.assertEqual(command[command.index('--preset')+1], 'quick')
        self.assertNotIn('--ball-map-policy', command)
        with self.assertRaises(ValueError):
            launcher.commands_for_run('quick', 'fixed-step-audit', Path('results/test'), resume=True)

    def test_map_policy_reaches_production_not_reference_generation(self):
        commands = launcher.commands_for_run('paper', 'logistic-regression', Path('results/test'), ball_map_policy='vanilla')
        self.assertNotIn('--ball-map-policy', commands[0])
        self.assertEqual(commands[1][commands[1].index('--ball-map-policy')+1], 'vanilla')
        command, = launcher.commands_for_run('paper', 'all', Path('results/test'), ball_map_policy='vanilla')
        self.assertEqual(command[command.index('--ball-map-policy')+1], 'vanilla')
        with self.assertRaises(ValueError):
            launcher.commands_for_run('paper', 'all', Path('results/test'), ball_map_policy='sqrt-d')

    def test_logistic_reference_precedes_sampling_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'logistic-reference').mkdir()
            result = root/'logistic-regression'
            result.mkdir()
            (result/'manifest.json').write_text('{}')
            workers = min(2, cpu_info()['capacity'])
            commands = launcher.commands_for_run('paper', 'logistic-regression', root, resume=True, workers=workers)
            self.assertEqual(len(commands), 2)
            self.assertIn(str(launcher.ROOT/'prepare_reference.py'), commands[0])
            self.assertIn(str(launcher.ROOT/'run_experiment.py'), commands[1])
            for command in commands:
                self.assertIn('--resume', command)
                self.assertEqual(command[command.index('--preset')+1], 'paper')
                self.assertEqual(command[command.index('--workers')+1], str(workers))
            self.assertEqual(commands[1][commands[1].index('--reference-dir')+1], str(root/'logistic-reference'))

    def test_replot_does_not_prepare_reference_or_start_new_sampling(self):
        command, = launcher.commands_for_run('paper', 'logistic-regression', Path('results/test'), plot_only=True)
        self.assertIn('--plot-only', command)
        self.assertNotIn('--preset', command)
        self.assertNotIn(str(launcher.ROOT/'prepare_reference.py'), command)

    def test_all_uses_complete_pipeline(self):
        command, = launcher.commands_for_run('paper', 'all', Path('results/test'), resume=True)
        self.assertIn(str(launcher.ROOT/'scripts/reproduce_all.py'), command)
        self.assertIn('--resume', command)
        self.assertEqual(command[command.index('--preset')+1], 'paper')

    def test_unknown_options_are_rejected(self):
        for preset, experiment in [('unknown', 'all'), ('quick', '../anything.py')]:
            with self.assertRaises(ValueError):
                launcher.commands_for_run(preset, experiment, Path('results/test'))

    def test_resume_uses_saved_settings_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, 'RUNS', Path(tmp)), \
                patch.object(launcher, 'environment_ready', return_value=True):
            controller = launcher.Controller()
            folder = Path(tmp)/'saved-paper-run'
            folder.mkdir()
            workers = min(2, cpu_info()['capacity'])
            (folder/'launcher-run.json').write_text(json.dumps({'preset': 'paper', 'experiment': 'garch11', 'workers': workers,
                                                              'ball_map_policy': 'vanilla'}))
            result = folder/'garch11'
            result.mkdir()
            (result/'manifest.json').write_text('{}')
            with patch.object(controller, 'run_command') as execute, \
                    patch.object(controller, 'start_job', side_effect=lambda work, status: work()):
                controller.run(preset='quick', experiment='all', run_id=folder.name)
                command = execute.call_args.args[0]
                self.assertIn('garch11', command)
                self.assertIn('--resume', command)
                self.assertEqual(command[command.index('--preset')+1], 'paper')
                self.assertEqual(command[command.index('--workers')+1], str(workers))
                self.assertEqual(command[command.index('--ball-map-policy')+1], 'vanilla')
                for invalid in ('..', '.', '../saved-paper-run', '/tmp'):
                    with self.assertRaises(ValueError):
                        controller.run(run_id=invalid)


if __name__ == '__main__':
    unittest.main()
