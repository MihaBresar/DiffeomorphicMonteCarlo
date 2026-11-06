"""Reported-study UI uses bundled settings and keeps earlier recipes separate."""
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import START_HERE as launcher
import run_experiment


def handler_for(controller, path, data=None):
    cls=launcher.make_handler(controller,'token')
    handler=cls.__new__(cls)
    payload=json.dumps(data or {}).encode()
    handler.server=SimpleNamespace(server_port=8000)
    handler.headers={'Host':'127.0.0.1:8000','X-Launcher-Token':'token','Content-Length':str(len(payload))}
    handler.path=path
    handler.rfile=BytesIO(payload)
    handler.respond=Mock()
    return handler


class ReportedStudyUiTests(unittest.TestCase):
    def test_file_open_hides_earlier_recipe_and_initializes_description(self):
        from html.parser import HTMLParser
        class MapParagraph(HTMLParser):
            attrs=None
            def handle_starttag(self,tag,attrs):
                if dict(attrs).get('id')=='map-description':self.attrs=dict(attrs)
        page=(launcher.ROOT/'launcher/index.html').read_text(encoding='utf-8')
        paragraph=MapParagraph();paragraph.feed(page)
        self.assertIn('hidden',paragraph.attrs)
        # Initialization must run for direct file previews as well as HTTP.
        self.assertIn('describe();\nlet timer;if(token)',page)

    def test_commands_keep_reported_study_distinct_and_forward_recorded_batches(self):
        for experiment in launcher.REPORTED_COMPARISONS:
            for target in launcher.COMPARISON_TARGETS:
                for preset in launcher.PRESETS:
                    command,=launcher.commands_for_run(preset,experiment,Path('unused'),comparison_target=target)
                    self.assertEqual(command[2],str(launcher.ROOT/'scripts/reproduce_reported_study.py'))
                    self.assertEqual(command[command.index('--target')+1],target)
                    self.assertEqual(command[command.index('--preset')+1],preset)
                    self.assertEqual(command[command.index('--comparison')+1],'numba' if experiment.endswith('-numba') else 'jax')
                    self.assertEqual(command[command.index('--seed-batch')+1],'22')
                    self.assertEqual(command[command.index('--output-dir')+1],str(Path('unused')/experiment))
                    for option in ('--vi','--ball-map-policy','--dcs-fixed-move','--optimizer-steps','--warmup'):
                        self.assertNotIn(option,command)
                for batch in (0,20,21,22,999):
                    command,=launcher.commands_for_run('paper',experiment,Path('unused'),seed_batch=batch)
                    self.assertEqual(command[command.index('--seed-batch')+1],str(batch))
                with self.assertRaisesRegex(ValueError,'start from scratch'):
                    launcher.commands_for_run('paper',experiment,Path('unused'),resume=True)

    def test_default_controller_and_http_record_loaded_maps_and_seed_22(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(launcher,'RUNS',Path(directory)), \
                patch.object(launcher,'environment_ready',return_value=True):
            controller=launcher.Controller()
            for via_http in (False,True):
                with patch.object(controller,'run_command') as execute, \
                        patch.object(controller,'start_job',side_effect=lambda work,status:work()):
                    if via_http:
                        handler=handler_for(controller,'/run')
                        handler.do_POST()
                        handler.respond.assert_called_once_with({'ok':True})
                    else:controller.run()
                folder=Path(directory)/controller.current_run
                request=json.loads((folder/'launcher_request.json').read_text())
                self.assertEqual(request,json.loads((folder/'launcher-run.json').read_text()))
                self.assertEqual(request['experiment'],'reported-zero-centre')
                self.assertEqual(request['seed_batch'],22)
                self.assertEqual(request['map_source'],'bundled_reported_study')
                self.assertFalse(request['fitting_performed'])
                self.assertNotIn('vi',request)
                self.assertNotIn('ball_map_policy',request)
                self.assertIn(str(launcher.ROOT/'scripts/reproduce_reported_study.py'),execute.call_args.args[0])

    def test_reported_http_rejects_extra_parameters_before_starting(self):
        controller=Mock()
        for extra in ({'vi':'on'},{'ball_map_policy':'mobius'},{'h':.15},
                      {'radius':1.},{'warmup':1},{'dcs_fixed_move':.07}):
            with self.subTest(extra=extra):
                handler=handler_for(controller,'/run',dict(experiment='reported-zero-centre',**extra))
                handler.do_POST()
                self.assertEqual(handler.respond.call_args.args[1],400)
                controller.run.assert_not_called()

    def test_public_http_accepts_all_current_experiment_seeds_and_rejects_retired_choices(self):
        controller=Mock()
        for experiment in launcher.EXPERIMENTS:
            base=22 if experiment in launcher.REPORTED_COMPARISONS+('all',) else 0
            for index in range(8):
                handler=handler_for(controller,'/run',dict(experiment=experiment,seed_batch=base+index))
                handler.do_POST()
                handler.respond.assert_called_once_with({'ok':True})
                self.assertEqual(controller.run.call_args.kwargs['seed_batch'],base+index)
        controller.reset_mock()
        for experiment in launcher.COMPARISONS+launcher.LEGACY_SCS_COMPARISONS:
            handler=handler_for(controller,'/run',dict(experiment=experiment))
            handler.do_POST()
            self.assertEqual(handler.respond.call_args.args[1],400)
        controller.run.assert_not_called()

    def test_seed_choice_is_recorded_for_every_current_experiment(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(launcher,'RUNS',Path(directory)), \
                patch.object(launcher,'environment_ready',return_value=True):
            controller=launcher.Controller()
            for experiment in launcher.EXPERIMENTS:
                batch=29 if experiment in launcher.REPORTED_COMPARISONS+('all',) else 7
                with patch.object(controller,'run_command'), \
                        patch.object(controller,'start_job',side_effect=lambda work,status:work()):
                    controller.run(experiment=experiment,seed_batch=batch)
                request=json.loads((Path(directory)/controller.current_run/'launcher_request.json').read_text())
                self.assertEqual(request['seed_choice'],8)
                self.assertEqual(request['seed_batch'],batch)
                if experiment=='all':self.assertEqual(request['generic_seed_batch'],7)

    def test_reported_cli_routes_only_recorded_inputs(self):
        module=ModuleType('scripts.reproduce_reported_study')
        module.main=Mock(return_value=73)
        for experiment in launcher.REPORTED_COMPARISONS:
            with patch.dict(sys.modules,{'scripts.reproduce_reported_study':module}):
                self.assertEqual(run_experiment.main([experiment,'--target','skew-t-500d',
                    '--preset','paper','--workers','2','--seed-batch','20','--output-dir','unused','--dry-run']),73)
            command=module.main.call_args.args[0]
            self.assertEqual(command[command.index('--comparison')+1],'numba' if experiment.endswith('-numba') else 'jax')
            for name,value in (('--target','skew-t-500d'),('--preset','paper'),('--workers','2'),('--seed-batch','20'),('--output-dir','unused')):
                self.assertEqual(command[command.index(name)+1],value)
            self.assertIn('--dry-run',command)
            self.assertNotIn('--vi',command)
            with patch.dict(sys.modules,{'scripts.reproduce_reported_study':module}):
                self.assertEqual(run_experiment.main([experiment,'--dry-run']),73)
            self.assertNotIn('--seed-batch',module.main.call_args.args[0])

    def test_fresh_cli_includes_original_and_calibrated_scs_in_both_backends(self):
        for experiment in launcher.REPORTED_COMPARISONS:
            with redirect_stdout(StringIO()) as output:
                self.assertEqual(run_experiment.main([experiment, '--dry-run']), 0)
            config = json.loads(output.getvalue())['configuration']
            self.assertIn(config['protocol'], (
                'timed-reported-zero-centre-v3', 'timed-reported-zero-centre-numba-v3'))
            self.assertEqual(config['algorithms'].count('SCS'), 1)
            self.assertEqual(config['algorithms'].count('SCS-calibrated'), 1)
            self.assertEqual(len(config['algorithms']), 6)

    def test_reported_cli_rejects_sampling_fitting_and_map_overrides(self):
        module=ModuleType('scripts.reproduce_reported_study')
        module.main=Mock()
        overrides=(['--warmup','1'],['--runs','5'],['--time-budget','1'],['--thin','2'],
            ['--optimizer-steps','2'],['--optimizer-samples','3'],['--vi','on'],
            ['--ball-map-policy','mobius'],['--ball-scale-policy','fit'],['--seed','1'],
            ['--scs-step-rule','fixed'],['--dcs-fixed-move','.025'],['--resume'])
        for experiment in launcher.REPORTED_COMPARISONS:
            for option in overrides:
                with self.subTest(experiment=experiment,option=option), \
                        patch.dict(sys.modules,{'scripts.reproduce_reported_study':module}), \
                        redirect_stderr(StringIO()),self.assertRaises(SystemExit) as error:
                    run_experiment.main([experiment,*option])
                self.assertEqual(error.exception.code,2)
                module.main.assert_not_called()

    def test_reported_plot_only_renders_saved_summary_without_sampling(self):
        for protocol in run_experiment.REPORTED_PROTOCOLS:
            with tempfile.TemporaryDirectory() as directory:
                folder=Path(directory)
                saved=json.dumps({'config':{'protocol':protocol},'preserved':'recorded settings'}).encode()
                (folder/'summary.json').write_bytes(saved)
                with patch('scripts.scs_100d_figures.render_report') as render, \
                        patch.object(run_experiment,'run') as sampling:
                    self.assertEqual(run_experiment.main(['--plot-only',str(folder)]),0)
                render.assert_called_once()
                sampling.assert_not_called()
                self.assertEqual((folder/'summary.json').read_bytes(),saved)

    def test_reported_replot_keeps_original_launcher_provenance(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(launcher,'RUNS',Path(directory)), \
                patch.object(launcher,'environment_ready',return_value=True):
            folder=Path(directory)/'saved';folder.mkdir()
            saved=json.dumps(dict(preset='reproduce',experiment='reported-zero-centre',workers=1,
                                  seed_batch=20,comparison_target='skew-t-500d',map_source='bundled_reported_study',fitting_performed=False)).encode()
            for name in ('launcher-run.json','launcher_request.json'):(folder/name).write_bytes(saved)
            controller=launcher.Controller()
            with patch.object(controller,'run_command') as execute, \
                    patch.object(controller,'start_job',side_effect=lambda work,status:work()):
                controller.run(run_id='saved',plot_only=True)
            command=execute.call_args.args[0]
            self.assertIn('--plot-only',command)
            self.assertNotIn('--seed-batch',command)
            for name in ('launcher-run.json','launcher_request.json'):self.assertEqual((folder/name).read_bytes(),saved)

    def test_retired_results_route_is_not_publicly_served(self):
        for path in ('/study-results/index.html','/study-results/summary.json',
                     '/study-results/../secret.json','/study-results/%2e%2e/secret.json'):
            handler=handler_for(Mock(),path)
            handler.do_GET()
            self.assertEqual(handler.respond.call_args.args[1],404)


if __name__=='__main__':unittest.main()
