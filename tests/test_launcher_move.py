"""Public launcher runs use preset steps and preserve saved-run provenance."""
from contextlib import redirect_stderr
from html.parser import HTMLParser
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import START_HERE as launcher
from scripts import reproduce_all


class LauncherMoveTests(unittest.TestCase):
    def controller(self):
        with patch.object(launcher, 'environment_ready', return_value=True):
            return launcher.Controller()

    def test_new_commands_use_protocol_presets(self):
        for experiment in launcher.EXPERIMENTS+launcher.LEGACY_SCS_COMPARISONS:
            with self.subTest(experiment=experiment):
                commands = launcher.commands_for_run('quick', experiment, Path('unused'))
                self.assertTrue(all('--dcs-fixed-move' not in command for command in commands))
                self.assertTrue(all('--dcs-step-rule' not in command for command in commands))

    def test_all_new_commands_reject_explicit_move_and_step_rule(self):
        for experiment in launcher.EXPERIMENTS+launcher.LEGACY_SCS_COMPARISONS:
            for extra in ({'dcs_fixed_move': .07}, {'dcs_fixed_move': .15},
                          {'dcs_step_rule': 'fixed'}, {'dcs_step_rule': 'acceptance'}):
                with self.subTest(experiment=experiment, extra=extra), \
                        self.assertRaisesRegex(ValueError, 'protocol preset step sizes'):
                    launcher.commands_for_run('paper', experiment, Path('unused'), **extra)

    def test_overrides_rejected_before_job_or_directory_creation(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(launcher, 'RUNS', Path(temp)):
            controller = self.controller()
            with patch.object(controller, 'start_job') as start:
                for move in (.07, .15, True, False, 0, -.1, float('nan'), float('inf'), -float('inf'), '0.15', [], {}, 10**1000):
                    with self.subTest(move=str(move)[:30]), self.assertRaisesRegex(ValueError, 'protocol preset step sizes'):
                        controller.run(dcs_fixed_move=move)
                for rule in ('fixed', 'acceptance', '', True, [], {}):
                    with self.subTest(rule=rule), self.assertRaisesRegex(ValueError, 'protocol preset step sizes'):
                        controller.run(dcs_step_rule=rule)
                start.assert_not_called()
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_new_requests_execute_without_step_overrides(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(launcher, 'RUNS', Path(temp)):
            controller = self.controller()
            for experiment in launcher.COMPARISONS+('all',):
                with self.subTest(experiment=experiment), patch.object(controller, 'run_command') as execute, \
                        patch.object(controller, 'start_job', side_effect=lambda work, status: work()):
                    controller.run(experiment=experiment)
                    folder = Path(temp)/controller.current_run
                    request = json.loads((folder/'launcher_request.json').read_text())
                    displayed = json.loads((folder/'launcher-run.json').read_text())
                    self.assertEqual(request, displayed)
                    self.assertNotIn('dcs_fixed_move', request)
                    self.assertNotIn('dcs_step_rule', request)
                    command = execute.call_args.args[0]
                    self.assertNotIn('--dcs-fixed-move', command)
                    self.assertNotIn('--dcs-step-rule', command)

    def test_http_rejects_removed_settings_including_null(self):
        controller = self.controller()
        handler_type = launcher.make_handler(controller, 'token')
        overrides = [{'dcs_fixed_move': value} for value in (.07, .15, True, '0.15', float('nan'), 0, None)]
        overrides += [{'dcs_step_rule': value} for value in ('fixed', 'acceptance', True, '', None)]
        for extra in overrides:
            with self.subTest(extra=extra), patch.object(controller, 'start_job') as start:
                payload = json.dumps(extra).encode()
                handler = handler_type.__new__(handler_type)
                handler.server = SimpleNamespace(server_port=8000)
                handler.headers = {'Host': '127.0.0.1:8000', 'X-Launcher-Token': 'token',
                                   'Content-Length': str(len(payload))}
                handler.path = '/run'
                handler.rfile = BytesIO(payload)
                handler.respond = Mock()
                handler.do_POST()
                start.assert_not_called()
                self.assertEqual(handler.respond.call_args.args[1], 400)
                self.assertIn('protocol preset step sizes', handler.respond.call_args.args[0]['error'])

    def test_replot_preserves_saved_settings_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(launcher, 'RUNS', Path(temp)):
            controller = self.controller()
            for experiment in launcher.COMPARISONS+('all',):
                folder = Path(temp)/experiment
                folder.mkdir()
                saved = json.dumps(dict(preset='paper', experiment=experiment, workers=1,
                                        dcs_fixed_move=.1, dcs_step_rule='fixed')).encode()
                for filename in ('launcher-run.json', 'launcher_request.json'):
                    (folder/filename).write_bytes(saved)
                with self.subTest(experiment=experiment), patch.object(controller, 'run_command') as execute, \
                        patch.object(controller, 'start_job', side_effect=lambda work, status: work()):
                    controller.run(run_id=folder.name, plot_only=True, dcs_fixed_move=.15)
                    command = execute.call_args.args[0]
                    self.assertIn('--plot-only', command)
                    self.assertNotIn('--dcs-fixed-move', command)
                    self.assertNotIn('--dcs-step-rule', command)
                    for filename in ('launcher-run.json', 'launcher_request.json'):
                        self.assertEqual((folder/filename).read_bytes(), saved)

    def test_all_seven_uses_protocol_presets(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(reproduce_all.subprocess, 'run') as execute, \
                patch('sys.argv', ['reproduce_all.py', '--output-root', temp]):
            reproduce_all.main()
            commands = [call.args[0] for call in execute.call_args_list]
            self.assertTrue(all('--dcs-fixed-move' not in command for command in commands))
            self.assertTrue(all('--dcs-step-rule' not in command for command in commands))
            self.assertEqual(sum('reported-zero-centre' in command for command in commands), 1)
            self.assertFalse(any('sampler-comparison' in command for command in commands))
            comparison, = [command for command in commands if 'reported-zero-centre' in command]
            self.assertEqual(comparison[comparison.index('--seed-batch')+1], '22')
            self.assertEqual(comparison[comparison.index('--target')+1], 'skew-t-100d')
            for unused in ('--reference-dir', '--vi', '--ball-map-policy'):
                self.assertNotIn(unused, comparison)
            self.assertEqual(len(commands), 9)

    def test_all_seven_forwards_selected_target_and_batch_to_comparison_only(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(reproduce_all.subprocess, 'run') as execute, \
                patch('sys.argv', ['reproduce_all.py', '--output-root', temp, '--target', 'skew-t-500d', '--seed-batch', '23']):
            reproduce_all.main()
            commands = [call.args[0] for call in execute.call_args_list]
            comparison, = [command for command in commands if 'reported-zero-centre' in command]
            self.assertEqual(comparison[comparison.index('--seed-batch')+1], '23')
            self.assertEqual(comparison[comparison.index('--target')+1], 'skew-t-500d')
            self.assertTrue(all('--seed-batch' not in command for command in commands if command is not comparison))

    def test_all_seven_replots_current_and_historical_comparisons_without_sampling(self):
        for comparison in ('reported-zero-centre', 'sampler-comparison', 'skew-t-100d-fixed'):
            with self.subTest(comparison=comparison), tempfile.TemporaryDirectory() as temp:
                (Path(temp)/comparison).mkdir()
                with patch.object(reproduce_all.subprocess, 'run') as execute, \
                        patch('sys.argv', ['reproduce_all.py', '--output-root', temp, '--plot-only']):
                    reproduce_all.main()
                commands = [call.args[0] for call in execute.call_args_list]
                replot = commands[2]
                self.assertEqual(replot[replot.index('--plot-only')+1], str(Path(temp).resolve()/comparison))
                self.assertEqual(len(commands), 8)
                self.assertTrue(all('--plot-only' in command for command in commands[:-1]))

    def test_all_seven_rejects_removed_flags_before_reference_preparation(self):
        for argument in ('--dcs-fixed-move=.07', '--dcs-fixed-move=.15', '--dcs-fixed-move=nan',
                         '--dcs-step-rule=fixed', '--dcs-step-rule=acceptance'):
            with self.subTest(argument=argument), patch.object(reproduce_all.subprocess, 'run') as execute, \
                    patch('sys.argv', ['reproduce_all.py', argument]), \
                    redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                reproduce_all.main()
            execute.assert_not_called()

    def test_launcher_has_no_manual_step_input(self):
        class Inputs(HTMLParser):
            def __init__(self): super().__init__(); self.values = {}
            def handle_starttag(self, tag, attrs):
                if tag in ('input', 'select'):
                    values = dict(attrs)
                    self.values[values.get('id')] = values
        parser = Inputs()
        parser.feed((launcher.ROOT/'launcher/index.html').read_text(encoding='utf-8'))
        self.assertNotIn('dcs-fixed-move', parser.values)
        self.assertNotIn('dcs-step-rule', parser.values)


if __name__ == '__main__':
    unittest.main()
