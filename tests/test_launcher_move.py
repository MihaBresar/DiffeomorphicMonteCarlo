"""Shared moves reach only current comparisons and preserve run provenance."""
from html.parser import HTMLParser
from io import BytesIO
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

    def test_optional_move_preserves_existing_defaults(self):
        for experiment in launcher.EXPERIMENTS+launcher.LEGACY_SCS_COMPARISONS:
            with self.subTest(experiment=experiment):
                commands = launcher.commands_for_run('quick', experiment, Path('unused'))
                self.assertTrue(all('--dcs-fixed-move' not in command for command in commands))

    def test_move_reaches_both_comparisons_and_all_seven_only(self):
        for experiment in launcher.COMPARISONS+('all',):
            for move in (.07, .15, .1, .025, 1):
                with self.subTest(experiment=experiment, move=move):
                    command, = launcher.commands_for_run('paper', experiment, Path('unused'), dcs_fixed_move=move)
                    self.assertEqual(float(command[command.index('--dcs-fixed-move')+1]), move)
        unsupported = set(launcher.EXPERIMENTS+launcher.LEGACY_SCS_COMPARISONS)-set(launcher.COMPARISONS+('all',))
        for experiment in unsupported:
            with self.subTest(experiment=experiment), self.assertRaisesRegex(ValueError, 'only for current'):
                launcher.commands_for_run('quick', experiment, Path('unused'), dcs_fixed_move=.15)

    def test_invalid_move_rejected_before_job_or_directory_creation(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(launcher, 'RUNS', Path(temp)):
            controller = self.controller()
            with patch.object(controller, 'start_job') as start:
                for move in (True, False, 0, -.1, float('nan'), float('inf'), -float('inf'), '0.15', [], {}, 10**1000):
                    with self.subTest(move=str(move)[:30]), self.assertRaisesRegex(ValueError, 'positive finite'):
                        controller.run(dcs_fixed_move=move)
                start.assert_not_called()
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_new_requests_record_and_execute_the_selected_move(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(launcher, 'RUNS', Path(temp)):
            controller = self.controller()
            for experiment in launcher.COMPARISONS+('all',):
                with self.subTest(experiment=experiment), patch.object(controller, 'run_command') as execute, \
                        patch.object(controller, 'start_job', side_effect=lambda work, status: work()):
                    controller.run(experiment=experiment, dcs_fixed_move=.025)
                    folder = Path(temp)/controller.current_run
                    request = json.loads((folder/'launcher_request.json').read_text())
                    displayed = json.loads((folder/'launcher-run.json').read_text())
                    self.assertEqual(request, displayed)
                    self.assertEqual(request['dcs_fixed_move'], .025)
                    command = execute.call_args.args[0]
                    self.assertEqual(command[command.index('--dcs-fixed-move')+1], '0.025')

    def test_http_keeps_value_types_and_controller_validates_them(self):
        controller = self.controller()
        handler_type = launcher.make_handler(controller, 'token')
        for value, valid in ((.15, True), (True, False), ('0.15', False), (float('nan'), False), (0, False)):
            with self.subTest(value=value), patch.object(controller, 'start_job') as start:
                payload = json.dumps({'dcs_fixed_move': value}).encode()
                handler = handler_type.__new__(handler_type)
                handler.server = SimpleNamespace(server_port=8000)
                handler.headers = {'Host': '127.0.0.1:8000', 'X-Launcher-Token': 'token',
                                   'Content-Length': str(len(payload))}
                handler.path = '/run'
                handler.rfile = BytesIO(payload)
                handler.respond = Mock()
                handler.do_POST()
                if valid:
                    start.assert_called_once()
                    handler.respond.assert_called_once_with({'ok': True})
                else:
                    start.assert_not_called()
                    self.assertEqual(handler.respond.call_args.args[1], 400)
                    self.assertIn('positive finite', handler.respond.call_args.args[0]['error'])

    def test_replot_preserves_saved_settings_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(launcher, 'RUNS', Path(temp)):
            controller = self.controller()
            for experiment in launcher.COMPARISONS+('all',):
                folder = Path(temp)/experiment
                folder.mkdir()
                saved = json.dumps(dict(preset='paper', experiment=experiment, workers=1,
                                        dcs_fixed_move=.1)).encode()
                for filename in ('launcher-run.json', 'launcher_request.json'):
                    (folder/filename).write_bytes(saved)
                with self.subTest(experiment=experiment), patch.object(controller, 'run_command') as execute, \
                        patch.object(controller, 'start_job', side_effect=lambda work, status: work()):
                    controller.run(run_id=folder.name, plot_only=True, dcs_fixed_move=.15)
                    command = execute.call_args.args[0]
                    self.assertIn('--plot-only', command)
                    self.assertNotIn('--dcs-fixed-move', command)
                    for filename in ('launcher-run.json', 'launcher_request.json'):
                        self.assertEqual((folder/filename).read_bytes(), saved)

    def test_all_seven_forwards_move_only_to_its_comparison(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(reproduce_all.subprocess, 'run') as execute, \
                patch('sys.argv', ['reproduce_all.py', '--output-root', temp, '--dcs-fixed-move', '.15']):
            reproduce_all.main()
            commands = [call.args[0] for call in execute.call_args_list]
            with_move = [command for command in commands if '--dcs-fixed-move' in command]
            self.assertEqual(len(with_move), 1)
            self.assertIn('sampler-comparison', with_move[0])
            self.assertEqual(float(with_move[0][with_move[0].index('--dcs-fixed-move')+1]), .15)
            self.assertEqual(len(commands), 9)

    def test_all_seven_rejects_invalid_move_before_reference_preparation(self):
        for value in ('0', '-.1', 'nan', 'inf'):
            with self.subTest(value=value), patch.object(reproduce_all.subprocess, 'run') as execute, \
                    patch('sys.argv', ['reproduce_all.py', '--dcs-fixed-move='+value]), \
                    self.assertRaises(SystemExit):
                reproduce_all.main()
            execute.assert_not_called()

    def test_move_input_keeps_required_precision(self):
        class Inputs(HTMLParser):
            def __init__(self): super().__init__(); self.values = {}
            def handle_starttag(self, tag, attrs):
                if tag == 'input':
                    values = dict(attrs)
                    self.values[values.get('id')] = values
        parser = Inputs()
        parser.feed((launcher.ROOT/'launcher/index.html').read_text(encoding='utf-8'))
        field = parser.values['dcs-fixed-move']
        self.assertEqual(field['type'], 'number')
        self.assertEqual(field['value'], '0.07')
        self.assertEqual(field['step'], 'any')
        self.assertIn('required', field)


if __name__ == '__main__':
    unittest.main()
