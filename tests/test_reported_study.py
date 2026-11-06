"""Frozen study inputs, fresh execution settings, and honest fit provenance."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dcs_bench import reported_study as study
from dcs_bench.references import radial_truth
from dcs_bench.targets import Target
from dcs_bench.timed_radial import _initial_proposal_scale, resolve_step_tuning
from scripts import reproduce_reported_study as cli
from scripts import reproduce_scs_100d as runner


class ReportedStudyTests(unittest.TestCase):
    def test_assets_match_the_original_bytes_and_author_fit(self):
        for d in (100, 500):
            with self.subTest(dimension=d):
                config = study.configuration(target=f'skew-t-{d}d')
                saved = json.loads((study.ASSETS / f'full_budget_{d}d.json').read_text(encoding='utf-8'))
                raw = (study.ASSETS / f'full_budget_{d}d.json').read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), study.FROZEN[d]['sha256'])
                self.assertEqual(len(raw), study.FROZEN[d]['bytes'])
                self.assertNotIn(b'/Users/', raw)
                prepared = study.load_prepared(config)
                self.assertEqual(prepared['maps']['sphere'], saved['maps']['sphere'])
                self.assertEqual(prepared['provenance']['original_fitted_maps'], saved['maps'])
                self.assertEqual(prepared['provenance']['original_fitting'], saved['fitting'])
                self.assertTrue(all(row['exact_equal'] for row in
                    saved['direct_author_verification']['checked_fields'].values()))
                for family in ('ball', 'sphere'):
                    current, original = prepared['fitting'][family], saved['fitting'][family]
                    self.assertEqual(current['loss'], original['loss'])
                    self.assertEqual(current['seconds'], 0.)
                    self.assertEqual(current['original_fit_seconds'], original['seconds'])
                    self.assertTrue(current['map_reused_without_refitting'])
                    self.assertFalse(current['fit_constrained_mu_zero'])
                    self.assertFalse(current['fit_constrained_delta_zero'])
                for method, mapping in runner.effective_method_maps(config, prepared['maps']).items():
                    if method not in ('SCS', 'SCS-calibrated'):
                        self.assertEqual(mapping, prepared['maps']['ball'])
                        self.assertEqual(mapping['scale'], study.MATCHED[d]['DCS-BW']['R'])
                        self.assertEqual(mapping['beta'], 2.)
                        np.testing.assert_array_equal(mapping['mu'], np.zeros(d))
                        np.testing.assert_array_equal(mapping['delta'], np.zeros(d))
                self.assertIn('not identical', prepared['provenance']['development_provenance']['scope'])

    def test_asset_tampering_is_rejected_without_fitting(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in ('manifest.json', 'full_budget_100d.json', 'joint_calibration.json', 'matched_calibration.json'):
                (directory / name).write_bytes((study.ASSETS / name).read_bytes())
            with patch.object(study, 'ASSETS', directory):
                study.configuration()
                with (directory / 'full_budget_100d.json').open('ab') as handle:
                    handle.write(b' ')
                with self.assertRaisesRegex(ValueError, 'hash or length'):
                    study.configuration()

    def test_presets_keep_original_maps_and_fixed_steps_for_every_backend(self):
        for backend in ('jax', 'numba'):
            for target, dimension in study.TARGETS.items():
                for preset, budget in study.PRESETS.items():
                    with self.subTest(backend=backend, target=target, preset=preset):
                        config = study.configuration(preset, comparison=backend, target=target)
                        self.assertEqual(config['runs'], budget['runs'])
                        self.assertEqual(config['time_budget'], budget['time_budget'])
                        self.assertEqual(config['warmup'], 1500)
                        self.assertEqual(config['thin'], 50)
                        self.assertEqual(config['seed_batch'], 22)
                        self.assertEqual(config['seed'], 22_901_001)
                        self.assertEqual(config['optimizer_steps'], 1000)
                        self.assertEqual(config['optimizer_samples'], 2000)
                        self.assertEqual(config['algorithms'], study.METHODS)
                        self.assertEqual(config['threshold'], 50. * math.sqrt(dimension / 100.))
                        self.assertTrue(config['ess_enabled'])
                        self.assertFalse(config['coordinate_metrics'])
                        for method in study.METHODS:
                            settings = runner.method_execution_config(config, method)
                            policy = ('author_scs_fixed_01' if method == 'SCS' else 'fixed_scs_move_v1'
                                      if method == 'SCS-calibrated' else 'fixed_dcs_move_v1')
                            self.assertEqual(resolve_step_tuning(settings), policy)
                            scale = (.1 if method == 'SCS' else study.MATCHED[dimension]['SCS']['h']
                                     if method == 'SCS-calibrated' else study.MATCHED[dimension]['DCS-BW']['h'] /
                                     (math.sqrt(dimension + 2) if method == 'DCS-RWM' else 1))
                            self.assertEqual(_initial_proposal_scale(settings), scale)

    def test_recorded_seed_batches_change_only_streams(self):
        first = study.configuration('reproduce', seed_batch=20)
        second = study.configuration('reproduce', seed_batch=21)
        self.assertEqual(first['seed'], 20_901_001)
        self.assertEqual(second['seed'], 21_901_001)
        self.assertEqual({k: v for k, v in first.items() if k not in ('seed', 'seed_batch')},
                         {k: v for k, v in second.items() if k not in ('seed', 'seed_batch')})
        self.assertEqual(study.load_prepared(first), study.load_prepared(second))

    def test_manual_scientific_overrides_are_not_available(self):
        for extra in ({'dcs_fixed_move': .07}, {'dcs_step_rule': 'acceptance'}, {'vi': 'off'},
                      {'optimizer_steps': 2}, {'runs': 1}, {'seed': 1}, {'scs_step_rule': 'acceptance'}):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'frozen'):
                study.configuration(overrides=extra)
        for batch in (-1, 1000, True, 1.5, '22'):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                study.configuration(seed_batch=batch)

    def test_forged_config_is_rejected_before_output_creation(self):
        original = study.configuration()
        for extra in ({'warmup': 10}, {'dcs_fixed_move': .07}, {'vi': 'off'},
                      {'scs_fixed_move': .1}, {'threshold': 25.}, {'seed': 1},
                      {'algorithms': ['SCS']}, {'step_scale_ceiling': 2.}):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / 'must-not-exist'
                with self.assertRaises(ValueError):
                    study.run(dict(original, **extra), output)
                self.assertFalse(output.exists())

    def test_prepared_maps_are_independent_copies_and_reports_validate_provenance(self):
        config = study.configuration()
        prepared = study.load_prepared(config)
        summary = dict(config=config, maps=prepared['maps'], fitting=prepared['fitting'],
                       method_maps=runner.effective_method_maps(config, prepared['maps']),
                       prepared_input_provenance=prepared['provenance'])
        study.validate_reported_summary(summary)
        damaged = copy.deepcopy(summary)
        damaged['maps']['sphere']['scale'] *= 2.
        with self.assertRaisesRegex(ValueError, 'maps or fitting'):
            study.validate_reported_summary(damaged)
        prepared['maps']['ball']['mu'][0] = 1.
        self.assertEqual(study.load_prepared(config)['maps']['ball']['mu'][0], 0.)

    def test_runner_reuses_prepared_inputs_without_any_fitting(self):
        config = study.configuration()
        prepared = study.load_prepared(config)
        _, truth, _ = radial_truth(Target(config['target']), config['threshold'])
        records = {method: [dict(chain=i, seed=config['seed'] + i, iterations=300,
            discarded_iterations=100, estimation_iterations=200, seconds=.05, acceptance=.5,
            final_scale=_initial_proposal_scale(runner.method_execution_config(config, method)),
            mean_all_iterations=(truth + np.array([.001, .1]) * (-1 if i else 1)).tolist())
            for i in range(2)] for method in config['algorithms']}
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, 'prepare') as fit, \
                patch.object(runner, 'run_chains', return_value=records) as sample, \
                patch('scripts.scs_100d_figures.render_report') as render:
            output = study.run(config, Path(temp) / 'reported')
            fit.assert_not_called()
            self.assertEqual(sample.call_args.args[2], prepared['maps'])
            render.assert_called_once()
            manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['prepared_input_provenance'], prepared['provenance'])
            self.assertGreaterEqual(manifest['prepared_input_load_seconds'], 0.)
            self.assertEqual(summary['prepared_input_load_seconds'], manifest['prepared_input_load_seconds'])
            study.validate_reported_summary(summary)
            self.assertIn('scripts/reproduce_reported_study.py', manifest['source_hashes'])
            for case in summary['cases'].values():
                np.testing.assert_allclose(case['mse'], [1e-6, .01])
                self.assertEqual(case['total_estimation_iterations'], 400)

    def test_direct_runner_requires_exact_prepared_inputs(self):
        config = study.configuration()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'must-not-exist'
            with self.assertRaisesRegex(ValueError, 'exact bundled'):
                runner.run(config, output)
            damaged = study.load_prepared(config)
            damaged['maps']['ball']['scale'] = 1.
            with self.assertRaisesRegex(ValueError, 'exact bundled'):
                runner.run(config, output, prepared_inputs=damaged)
            self.assertFalse(output.exists())

    def test_cli_dry_run_is_portable_and_manual_flags_rejected(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.main(['--dry-run', '--target', 'skew-t-500d', '--seed-batch', '21']), 0)
        config = json.loads(output.getvalue())['configuration']
        self.assertEqual(config['seed'], 21_901_001)
        self.assertEqual(config['dimension'], 500)
        self.assertFalse(Path(config['reported_map_asset']).is_absolute())
        self.assertNotIn('/Users/', json.dumps(config))
        for argument in ('--dcs-fixed-move=.15', '--vi=off', '--optimizer-steps=10', '--warmup=10'):
            with self.subTest(argument=argument), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                cli.main([argument, '--dry-run'])


if __name__ == '__main__':
    unittest.main()
