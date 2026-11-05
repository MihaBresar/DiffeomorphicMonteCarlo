"""Cross-component fairness contracts for the two selectable-target suites."""
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import special, stats

from dcs_bench import timed_radial
from dcs_bench.transforms import BallMap, SphereMap
from scripts import reproduce_comparison as suite
from scripts import reproduce_scs_100d as runner

class ComparisonSuiteTests(unittest.TestCase):

    def test_author_vi_presets_and_explicit_budget_overrides(self):
        for preset, scs_steps, dcs_steps, draws in (
                ('quick',20,20,128),('reproduce',400,400,500),('paper',1000,1000,2000)):
            config = suite.configuration(preset)
            self.assertEqual(config['scs_vi'], dict(optimizer_steps=scs_steps,optimizer_samples=draws))
            self.assertEqual(config['optimizer_steps'],dcs_steps)
            self.assertEqual(config['sphere_map_policy'],'author-paper-vi')
        config = suite.configuration('paper',overrides={'optimizer_steps':7,'optimizer_samples':31})
        self.assertEqual(config['scs_vi'],dict(optimizer_steps=7,optimizer_samples=31))
        self.assertEqual(config['optimizer_steps'],7)

    def test_explicit_scs_step_rules_keep_other_methods_unchanged(self):
        tuned=suite.configuration('quick',overrides={'scs_step_rule':'acceptance'})
        fixed=suite.configuration('quick',overrides={'scs_step_rule':'fixed'})
        self.assertEqual(fixed['step_tuning_policies']['SCS'],'author_scs_fixed_01')
        for method in ('DCS-Billiard','DCS-BW-reflect'):
            self.assertEqual(fixed['step_tuning_policies'][method],tuned['step_tuning_policies'][method])
        self.assertIn('no step adaptation',fixed['tuning_note'])
        with self.assertRaisesRegex(ValueError,'scs_step_rule'):
            suite.configuration('quick',overrides={'scs_step_rule':'best-MSE'})

    def test_fit_seed_override_matches_reported_and_executed_dcs_fit(self):
        config = suite.configuration('quick', overrides={'fit_seed': 73})
        self.assertIn('DCS: fixed-cloud reverse KL, seed 73,', config['fitting_policy'])
        self.assertIn('SCS: unchanged author VI, fixed base batch, seed 0,', config['fitting_policy'])
        with patch.object(runner, 'fit_map', return_value=(
                BallMap(100), {'seed': 73, 'steps': 20, 'training_draws': 128})) as dcs_fit, \
                patch('dcs_bench.scs_paper.fit', return_value=(
                    SphereMap(100).parameters(), {'seed': 0, 'steps': 20, 'training_draws': 128})) as scs_fit, \
                redirect_stdout(StringIO()):
            _, fitting = runner.prepare(config)
        self.assertEqual(dcs_fit.call_args.args[2:], ('ball', 73))
        self.assertEqual(fitting['ball']['seed'], 73)
        self.assertEqual(fitting['sphere']['seed'], 0)
        self.assertNotIn('fit_seed', scs_fit.call_args.args[0])

    def test_reflected_bw_retains_existing_pilot_defaults(self):
        from scripts import reproduce_boundary_100d as boundary
        previous = boundary.configuration('paper')
        for backend in ('jax', 'numba'):
            config = suite.configuration('paper', comparison=backend)
            method_config = dict(config, algorithm='DCS-BW-reflect')
            self.assertEqual(timed_radial.resolve_step_tuning(method_config),
                             'ball_esjd_pilot_v1')
            for key in ('initial_scale', 'warmup', 'seed', 'burnin_fraction'):
                self.assertEqual(config[key], previous[key])
            self.assertNotIn('step_scale_ceilings', config)

    def test_suite_configuration_selects_target_methods_beta_and_shared_resources(self):
        expected_methods = {backend: ['DCS-RWM', 'DCS-BW', 'SCS', 'DCS-BW-reflect', 'DCS-Billiard']
                            for backend in ('jax', 'numba')}
        target_cases = (('skew-t-100d', 'skew_t', 100, 2.),
                        ('skew-t-500d', 'skew_t', 500, 2.))
        for backend, methods in expected_methods.items():
            for target_id, kind, dimension, beta in target_cases:
                for vi in ('on',):
                    with self.subTest(backend=backend, target=target_id, vi=vi):
                        config = suite.configuration('paper', comparison=backend, target=target_id, vi=vi)
                        self.assertEqual(config['algorithms'], methods)
                        self.assertEqual(config['target']['kind'], kind)
                        self.assertEqual(config['target']['dimension'], dimension)
                        self.assertEqual(config['dimension'], dimension)
                        self.assertEqual(config['beta'], beta)
                        self.assertEqual(config['dcs_backend'], backend)
                        self.assertEqual(config['vi_enabled'], vi == 'on')
                        self.assertEqual(config['step_tuning_policies'], {'DCS-BW-reflect': 'ball_esjd_pilot_v1', 'DCS-Billiard': 'own_kernel_radial_pilot_v1', 'SCS': 'author_scs_fixed_01'})
                        self.assertNotIn('mixture_probability', config)
                        self.assertNotIn('initial_angular_scale', config)
                        self.assertNotIn('step_scale_ceilings', config)
                        self.assertEqual(config['runs'], 50)
                        self.assertEqual(config['warmup'], 1500)
                        self.assertEqual(config['time_budget'], 20.)
                        self.assertEqual(config['burnin_fraction'], 1/3)
                        self.assertEqual(config['optimizer_steps'], 1000)
                        self.assertEqual(config['optimizer_samples'], 2000)
                        self.assertEqual(config['target']['nu'], 3.)
                        self.assertEqual(config['target']['alpha'], {'0': 100., '1': -100.})
                        self.assertFalse(config['coordinate_metrics'])
                        self.assertNotIn('coordinate_threshold', config)
                        self.assertTrue(config['ess_enabled'])
                        self.assertEqual(config['reference_samples'], 5_000_000)
                        self.assertTrue(config['protocol'].endswith('-v10'))

    def test_vi_off_bypasses_fitting_for_both_map_families_without_skipping_warmup(self):
        for target_id in ('skew-t-100d', 'skew-t-500d'):
            # An explicit historical fixture exercises the retained low-level geometry.
            config = suite.configuration('quick', comparison='numba', target=target_id)
            config.update(vi='off', vi_enabled=False, ball_map_policy='vanilla', sphere_map_policy='vanilla')
            with self.subTest(target=target_id), patch.object(runner, 'fit_map') as fit, \
                    redirect_stdout(StringIO()):
                maps, metadata = runner.prepare(config)
                fit.assert_not_called()
                self.assertEqual(config['warmup'], 100)
                for family in ('ball', 'sphere'):
                    mapping, details = maps[family], metadata[family]
                    self.assertEqual(mapping['dimension'], config['dimension'])
                    self.assertEqual(mapping['scale'], 1.)
                    np.testing.assert_array_equal(mapping['mu'], 0.)
                    np.testing.assert_array_equal(mapping['delta' if family == 'ball' else 'observer'], 0.)
                    self.assertEqual(details['steps'], 0)
                    self.assertEqual(details['training_draws'], 0)
                    self.assertEqual(details['total_training_draws'], 0)
                    self.assertFalse(details['vi_enabled'])
                    self.assertIsNone(details['seed'])
                self.assertEqual(maps['ball']['beta'], config['beta'])
                self.assertEqual(maps['sphere']['latitude'], config['latitude'])

    def test_old_sphere_policy_still_dispatches_to_author_vi(self):
        config = suite.configuration('paper', comparison='jax', target='skew-t-500d', vi='on')
        config['sphere_map_policy'] = 'vi'
        calls = []

        def fake_fit(target, settings, family, seed):
            calls.append((target, dict(settings), family, seed))
            return BallMap(target.d, beta=settings['beta']), {
                'steps': settings['optimizer_steps'], 'training_draws': settings['optimizer_samples']}

        author_result = (SphereMap(500).parameters(), {'steps': 1000, 'training_draws': 2000, 'seed': 0})
        with patch.object(runner, 'fit_map', side_effect=fake_fit), \
                patch('dcs_bench.scs_paper.fit', return_value=author_result) as author, \
                redirect_stdout(StringIO()):
            maps, metadata = runner.prepare(config)
        self.assertEqual([call[2] for call in calls], ['ball'])
        self.assertEqual(calls[0][0].d, 500)
        self.assertEqual(calls[0][3], 42)
        author.assert_called_once_with({'target': config['target'], **config['scs_vi']})
        self.assertEqual(maps['ball']['beta'], 2.)
        self.assertEqual([metadata[f]['steps'] for f in ('ball', 'sphere')], [1000, 1000])
        self.assertEqual([metadata[f]['training_draws'] for f in ('ball', 'sphere')], [2000, 2000])

    def test_summarize_uses_selected_target_truth_instead_of_old_100d_constants(self):
        for target_id in ('skew-t-100d', 'skew-t-500d'):
            config = suite.configuration('quick', overrides={'runs': 2}, comparison='jax', target=target_id)
            dimension = config['dimension']
            mean_norm = np.exp(.5*np.log(3.)+special.gammaln((dimension+1)/2)+special.gammaln(1.)
                               -special.gammaln(dimension/2)-special.gammaln(1.5))
            truth = np.array([stats.f.sf(config['threshold']**2/dimension, dimension, 3), mean_norm])
            reference = dict(functionals=['tail_probability', 'mean_norm'], truth=truth.tolist(),
                             reference_mcse=[0., 0.], exact=[True, True], source='Analytic Student-t radial law')
            errors = np.array([.005, 1.])
            records = {}
            for method in config['algorithms']:
                records[method] = []
                for chain, direction in enumerate((-1, 1)):
                    count = (chain+1)*300
                    records[method].append(dict(chain=chain, seed=config['seed']+chain,
                        iterations=count, discarded_iterations=count//3, estimation_iterations=count*2//3,
                        seconds=.1, acceptance=.5, final_scale=.1,
                        mean_all_iterations=(truth+direction*errors).tolist()))
            with self.subTest(target=target_id):
                summary = runner.summarize(config, {}, {}, records, reference=reference)
                np.testing.assert_allclose(summary['truth'], truth, rtol=1e-12, atol=1e-12)
                self.assertEqual(summary['target_note'], config['target_note'])
                for case in summary['cases'].values():
                    np.testing.assert_allclose(case['mse'], errors**2, rtol=1e-11, atol=1e-12)
                    self.assertEqual(case['total_iterations'], 900)
                    self.assertEqual(case['total_estimation_iterations'], 600)
                self.assertEqual(len(summary['paired_mse_differences']), 8)
                self.assertEqual(summary['reference'], reference)

    def test_native_dispatch_reuses_same_jax_calibration_and_keeps_author_scs(self):
        from dcs_bench import numba_radial
        mapping = {'dimension': 100}
        calibration = SimpleNamespace(scale=.4, state=np.zeros(100))
        native = object()
        for method in ('DCS-BW', 'DCS-BW-reflect', 'SCS'):
            config = dict(suite.configuration('quick', comparison='numba'), algorithm=method)
            with self.subTest(method=method), \
                    patch.object(timed_radial, '_Matched', return_value=calibration) as calibrate, \
                    patch.object(numba_radial, 'NumbaRadial', return_value=native) as production:
                result = timed_radial._backend(mapping, config, 901001)
                calibrate.assert_called_once_with(mapping, config, 901001)
                if method == 'SCS':
                    production.assert_not_called()
                    self.assertIs(result, calibration)
                else:
                    production.assert_called_once_with(mapping, config, 901001, calibration)
                    self.assertIs(result, native)

    def test_rotation_dispatch_reuses_both_jax_calibrated_scales(self):
        from dcs_bench import numba_rotation, numba_radial
        mapping = {'dimension': 100}
        calibration = SimpleNamespace(scale=.4, angular_scale=2., state=np.zeros(100))
        native = object()
        config = dict(suite.configuration('quick', comparison='numba'), algorithm='DCS-BW-rotate')
        with patch.object(timed_radial, '_Matched', return_value=calibration) as calibrate, \
                patch.object(numba_rotation, 'NumbaRotation', return_value=native) as production, \
                patch.object(numba_radial, 'NumbaRadial') as legacy:
            result = timed_radial._backend(mapping, config, 901001)
            calibrate.assert_called_once_with(mapping, config, 901001)
            production.assert_called_once_with(mapping, config, 901001, calibration)
            legacy.assert_not_called()
            self.assertIs(result, native)

    def test_rotation_metadata_accounts_for_exact_shared_pilot_budget(self):
        # Seven transitions: three BW trials, then angular trials of 2, 1, 1.
        bw = np.array([[[.1, 1, .2, 1., 0], [.2, 1, .4, 1., 1], [.4, 1, .3, 1., 0]]])
        angular = np.array([[[.5, 2, .5, 1., 0], [1., 1, 1., 1., 1], [2., 1, .2, 0., 0]]])
        metadata = timed_radial._rotation_tuning_metadata((bw, angular), .2, 1., {'warmup': 7})
        self.assertEqual(metadata['tuning_pilot_iterations'], 7)
        self.assertEqual(metadata['bw_warmup_iterations'], 3)
        self.assertEqual(metadata['angular_warmup_iterations'], 4)
        self.assertEqual(sum(row['iterations'] for row in metadata['tuning_history']), 3)
        self.assertEqual(sum(row['iterations'] for row in metadata['angular_tuning_history']), 4)
        self.assertAlmostEqual(metadata['angular_angle'], np.pi/2)
        self.assertEqual(metadata['mixture_probability'], .5)

    def test_gaussian_rotation_dispatch_scale_units_and_shared_calibration(self):
        from dcs_bench import numba_rwm_rotation, numba_rotation
        config = dict(suite.configuration('quick', comparison='numba'), algorithm='DCS-RWM-rotate')
        expected = .1 / np.sqrt(config['dimension'] + 2)
        self.assertEqual(timed_radial.resolve_step_tuning(config), timed_radial.ROTATION_TUNING)
        self.assertAlmostEqual(timed_radial._initial_proposal_scale(config), expected)
        calibration = SimpleNamespace(scale=expected*4, angular_scale=2., state=np.zeros(100))
        mapping, native = {'dimension': 100}, object()
        with patch.object(timed_radial, '_Matched', return_value=calibration), \
                patch.object(numba_rwm_rotation, 'NumbaRWMRotation', return_value=native) as production, \
                patch.object(numba_rotation, 'NumbaRotation') as bw:
            self.assertIs(timed_radial._backend(mapping, config, 1901001), native)
            production.assert_called_once_with(mapping, config, 1901001, calibration)
            bw.assert_not_called()
        # A Gaussian-SD ceiling below the BW reference radius is admissible.
        self.assertEqual(timed_radial.resolve_step_scale_ceiling(dict(config,
            step_scale_ceilings={'DCS-RWM-rotate': .05})), .05)
        with self.assertRaises(ValueError):
            timed_radial.resolve_step_scale_ceiling(dict(config, step_scale_ceiling=expected/2))

    def test_gaussian_rotation_metadata_preserves_units_and_exact_budget(self):
        config = dict(suite.configuration('quick'), algorithm='DCS-RWM-rotate', warmup=7)
        sigma = timed_radial._initial_proposal_scale(config)
        reflection = np.array([[[sigma/2, 1, .2, 1., 0], [sigma, 1, .4, 1., 1], [sigma*2, 1, .3, 1., 0]]])
        angular = np.array([[[.5, 2, .5, 1., 0], [1., 1, 1., 1., 1], [2., 1, .2, 0., 0]]])
        info = timed_radial._rotation_tuning_metadata((reflection, angular), sigma, 1., config)
        self.assertEqual(info['initial_scale_reference_bw_radius'], .1)
        self.assertEqual(info['initial_gaussian_sigma'], sigma)
        self.assertAlmostEqual(info['gaussian_equivalent_bw_radius'], .1)
        self.assertEqual(info['scale_parameter'], 'Gaussian coordinate standard deviation')
        self.assertEqual(info['tuning_pilot_iterations'], 7)
        self.assertEqual(info['gaussian_warmup_iterations'], 3)
        self.assertEqual(info['angular_warmup_iterations'], 4)

if __name__ == '__main__':
    unittest.main()
