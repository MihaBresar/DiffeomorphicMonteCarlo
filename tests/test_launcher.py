"""Check GUI dispatch preserves the scientific pipeline and saved-run settings."""
import json
from io import BytesIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import START_HERE as launcher
from dcs_bench.cpu import cpu_info


class LauncherTests(unittest.TestCase):
    def test_single_scs_comparison_runs_fresh_pipeline(self):
        workers = min(2, cpu_info()['capacity'])
        for preset in ('quick', 'reproduce', 'paper'):
            with self.subTest(preset=preset):
                command, = launcher.commands_for_run(preset, 'skew-t-100d-fixed', Path('results/test'),
                                                      workers=workers, ball_map_policy='mobius')
                self.assertIn(str(launcher.ROOT/'scripts/reproduce_scs_100d.py'), command)
                self.assertEqual(command[command.index('--preset')+1], preset)
                self.assertEqual(command[command.index('--workers')+1], str(workers))
                self.assertEqual(command[command.index('--output-dir')+1],
                                 str(Path('results/test')/'skew-t-100d-fixed'))
                self.assertNotIn('--ball-map-policy', command)
        for options in ({'resume': True}, {'plot_only': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                launcher.commands_for_run('quick', 'skew-t-100d-fixed', Path('results/test'), **options)

    def test_diagnostic_audits_are_archived_from_launcher(self):
        for experiment in ('fixed-step-audit', 'symmetric-t-audit', 'skew-t-timed-audit', 'joint-r-step-audit'):
            with self.subTest(experiment=experiment), self.assertRaises(ValueError):
                launcher.commands_for_run('quick', experiment, Path('results/test'))

    def test_visible_menu_uses_short_names_and_removes_earlier_comparisons(self):
        from html.parser import HTMLParser
        class ExperimentMenu(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_menu = False
                self.options = []
                self.labels = []
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == 'select':
                    self.in_menu = attrs.get('id') == 'experiment'
                elif tag == 'option' and self.in_menu:
                    self.options.append(attrs)
            def handle_data(self, data):
                if self.in_menu:
                    self.labels.append(data)
            def handle_endtag(self, tag):
                if tag == 'select':
                    self.in_menu = False
        menu = ExperimentMenu()
        menu.feed((launcher.ROOT/'launcher/index.html').read_text(encoding='utf-8'))
        self.assertEqual({x['value'] for x in menu.options}, set(launcher.EXPERIMENTS))
        self.assertEqual([x['value'] for x in menu.options if 'selected' in x], ['reported-zero-centre'])
        self.assertEqual(menu.options[0]['value'], 'reported-zero-centre')
        self.assertEqual(menu.labels[:3], ['DCS vs original SCS', 'Numba extension', 'All seven experiments + posterior table'])
        self.assertTrue(set(launcher.COMPARISONS).isdisjoint(x['value'] for x in menu.options))
        self.assertTrue(set(launcher.LEGACY_SCS_COMPARISONS).isdisjoint(x['value'] for x in menu.options))
        target_menu = ExperimentMenu()
        target_menu.feed((launcher.ROOT/'launcher/index.html').read_text(encoding='utf-8')
                         .replace('id="experiment"', 'id="other-menu"')
                         .replace('id="comparison-target"', 'id="experiment"'))
        self.assertEqual([x['value'] for x in target_menu.options], ['skew-t-100d', 'skew-t-500d'])
        self.assertEqual(tuple(x['value'] for x in target_menu.options), launcher.COMPARISON_TARGETS)
        seed_menu = ExperimentMenu()
        seed_menu.feed((launcher.ROOT/'launcher/index.html').read_text(encoding='utf-8')
                       .replace('id="experiment"', 'id="other-menu"').replace('id="seed-batch"', 'id="experiment"'))
        self.assertEqual([x['value'] for x in seed_menu.options], [str(x) for x in range(22,31)]+['20','21'])
        self.assertEqual([x['value'] for x in seed_menu.options if 'selected' in x], ['22'])

    def test_seed_batches_route_without_changing_fitting_inputs(self):
        for experiment in launcher.SCS_COMPARISONS:
            for batch in (0, 1, 5, 6, 7, 8, 999):
                with self.subTest(experiment=experiment, batch=batch):
                    command, = launcher.commands_for_run('quick', experiment, Path('results/test'), seed_batch=batch)
                    if batch or experiment in launcher.REPORTED_COMPARISONS:
                        self.assertEqual(command[command.index('--seed-batch')+1], str(batch))
                    else:
                        self.assertNotIn('--seed-batch', command)
                    self.assertNotIn('--seed', command)
                    self.assertNotIn('--fit-seed', command)
        for experiment in set(launcher.EXPERIMENTS)-set(launcher.SEED_BATCH_EXPERIMENTS):
            with self.subTest(experiment=experiment):
                commands = launcher.commands_for_run('quick', experiment, Path('results/test'))
                self.assertTrue(all('--seed-batch' not in command for command in commands))
                with self.assertRaisesRegex(ValueError, 'only for the sampler comparisons'):
                    launcher.commands_for_run('quick', experiment, Path('results/test'), seed_batch=1)

    def test_invalid_seed_batches_fail_before_any_job_starts(self):
        with patch.object(launcher, 'environment_ready', return_value=True):
            controller = launcher.Controller()
        with patch.object(controller, 'start_job') as start:
            for batch in (True, False, None, '1', 1.0, 0.0, -1, 1000, [], {}):
                with self.subTest(batch=batch), self.assertRaisesRegex(ValueError, 'integer from 0 to 999'):
                    controller.run(seed_batch=batch)
            start.assert_not_called()

    def test_fresh_runs_record_batch_and_forward_it_to_sampling(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, 'RUNS', Path(tmp)), \
                patch.object(launcher, 'environment_ready', return_value=True):
            controller = launcher.Controller()
            for experiment in launcher.SCS_COMPARISONS:
                for batch in (0, 3):
                    with self.subTest(experiment=experiment, batch=batch), \
                            patch.object(controller, 'run_command') as execute, \
                            patch.object(controller, 'start_job', side_effect=lambda work, status: work()):
                        controller.run(experiment=experiment, seed_batch=batch)
                        metadata = json.loads((Path(tmp)/controller.current_run/'launcher-run.json').read_text())
                        self.assertEqual(metadata['seed_batch'], batch)
                        self.assertEqual(metadata['experiment'], experiment)
                        command = execute.call_args.args[0]
                        if batch or experiment in launcher.REPORTED_COMPARISONS:
                            self.assertEqual(command[command.index('--seed-batch')+1], str(batch))
                        else:
                            self.assertNotIn('--seed-batch', command)

    def test_fresh_runs_reserve_distinct_folders_when_clock_and_candidate_collide(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, 'RUNS', Path(tmp)), \
                patch.object(launcher, 'environment_ready', return_value=True), \
                patch.object(launcher, 'datetime') as clock, \
                patch.object(launcher.secrets, 'token_hex', side_effect=['occupied', 'first', 'second']):
            clock.now.return_value.strftime.return_value = 'same-clock'
            existing = Path(tmp)/'same-clock-quick'
            existing.mkdir()
            original = {'previous_run': 'must remain unchanged'}
            (existing/'launcher-run.json').write_text(json.dumps(original))
            (existing/'launcher.log').write_text('Previous log\n')
            (Path(tmp)/'same-clock-quick-occupied').mkdir()
            controller = launcher.Controller()
            folders = []
            for experiment, batch in [('sampler-comparison', 6), ('skew-t-100d-boundary', 0)]:
                with patch.object(controller, 'run_command') as execute, \
                        patch.object(controller, 'start_job', side_effect=lambda work, status: work()):
                    controller.run(experiment=experiment, seed_batch=batch)
                    folder = Path(tmp)/controller.current_run
                    folders.append(folder)
                    metadata = json.loads((folder/'launcher-run.json').read_text())
                    self.assertEqual(metadata['experiment'], experiment)
                    self.assertEqual(metadata['seed_batch'], batch)
                    command = execute.call_args.args[0]
                    self.assertEqual(command[command.index('--output-dir')+1], str(folder/experiment))
                    self.assertNotIn('--resume', command)
            self.assertEqual([folder.name for folder in folders],
                             ['same-clock-quick-first', 'same-clock-quick-second'])
            self.assertEqual(json.loads((existing/'launcher-run.json').read_text()), original)
            self.assertEqual((existing/'launcher.log').read_text(), 'Previous log\n')
            self.assertFalse(any((Path(tmp)/'same-clock-quick-occupied').iterdir()))

    def test_http_api_forwards_batch_without_coercing_invalid_types(self):
        controller = Mock()
        handler_type = launcher.make_handler(controller, 'test-token')
        for batch in (4, True, '2'):
            with self.subTest(batch=batch):
                payload = json.dumps({'seed_batch': batch}).encode()
                handler = handler_type.__new__(handler_type)
                handler.server = SimpleNamespace(server_port=8000)
                handler.headers = {'Host': '127.0.0.1:8000', 'X-Launcher-Token': 'test-token',
                                   'Content-Length': str(len(payload))}
                handler.path = '/run'
                handler.rfile = BytesIO(payload)
                handler.respond = Mock()
                handler.do_POST()
                forwarded = controller.run.call_args.kwargs['seed_batch']
                self.assertEqual(forwarded, batch)
                self.assertIs(type(forwarded), type(batch))

    def test_map_policy_reaches_production_not_reference_generation(self):
        commands = launcher.commands_for_run('paper', 'logistic-regression', Path('results/test'), ball_map_policy='mobius')
        self.assertNotIn('--ball-map-policy', commands[0])
        self.assertEqual(commands[1][commands[1].index('--ball-map-policy')+1], 'mobius')
        command, = launcher.commands_for_run('paper', 'all', Path('results/test'), ball_map_policy='mobius')
        self.assertEqual(command[command.index('--ball-map-policy')+1], 'mobius')
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
        command, = launcher.commands_for_run('paper', 'all', Path('results/test'), comparison_target='skew-t-500d')
        self.assertIn(str(launcher.ROOT/'scripts/reproduce_all.py'), command)
        self.assertNotIn('--resume', command)
        self.assertEqual(command[command.index('--seed-batch')+1], '22')
        self.assertEqual(command[command.index('--target')+1], 'skew-t-500d')
        self.assertEqual(command[command.index('--preset')+1], 'paper')
        with self.assertRaisesRegex(ValueError, 'starts from scratch'):
            launcher.commands_for_run('paper', 'all', Path('results/test'), resume=True)

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
                                                              'ball_map_policy': 'mobius'}))
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
                self.assertEqual(command[command.index('--ball-map-policy')+1], 'mobius')
                for invalid in ('..', '.', '../saved-paper-run', '/tmp'):
                    with self.assertRaises(ValueError):
                        controller.run(run_id=invalid)


if __name__ == '__main__':
    unittest.main()
