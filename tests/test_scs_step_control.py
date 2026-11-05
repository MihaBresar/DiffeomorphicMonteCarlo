"""Matched historical controls must reuse a validated primary without refitting."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import reproduce_scs_step_control as control
from scripts import reproduce_comparison as suite
from scripts import reproduce_scs_100d as runner
from dcs_bench.transforms import BallMap, SphereMap


SOURCES = {'dcs_bench/timed_radial.py': 'a'*64, 'scripts/scs_100d_figures.py': 'b'*64}
HARDWARE = {'capacity': 2, 'cpu_ids': [None, None]}


def write(path, value):
    path.write_text(json.dumps(value))


def primary_fixture(root, backend='jax', vi='on'):
    with patch.object(runner, 'cpu_info', return_value=HARDWARE):
        config = suite.configuration('quick', workers=2, overrides={'scs_step_rule':'acceptance'}, comparison=backend, vi=vi)
    d = config['dimension']
    maps = {'ball': BallMap(d, scale=1.7).parameters(), 'sphere': SphereMap(d, scale=.8).parameters()}
    fitting = {family: {'seconds': 1.25, 'steps': 20} for family in maps}
    method_maps = runner.effective_method_maps(config, maps)
    cases = {}
    for method in control.METHODS:
        cases[method] = {'chains': config['runs'], 'records': [dict(algorithm=method,
            chain=i, seed=config['seed']+i, warmup_iterations=config['warmup'],
            calibration_algorithm=method, tuning_policy='indicator_dual_averaging_0234',
            radial_tuning_pilot_iterations=config['warmup'],
            radial_tuning_history=[{'iterations': 13 if j<4 else 12} for j in range(8)],
            iterations=600, estimation_iterations=400, sampling_seconds=.06,
            adaptation_during_sampling=False) for i in range(config['runs'])]}
    files = {'configuration.json': config,
        'manifest.json': dict(status='complete', protocol=config['protocol'], config=config,
            source_hashes=SOURCES.copy(), method_maps=method_maps),
        'maps.json': dict(maps=maps, fitting=fitting, method_maps=method_maps),
        'method_maps.json': method_maps,
        'summary.json': dict(config=config, maps=maps, fitting=fitting,
                             method_maps=method_maps, cases=cases)}
    root.mkdir()
    for name, value in files.items(): write(root/name, value)
    return deepcopy(files)


class CalibrationControlTests(unittest.TestCase):
    def setUp(self):
        self.sources = patch.object(runner, 'source_hashes', return_value=SOURCES.copy())
        self.hardware = patch.object(runner, 'cpu_info', return_value=HARDWARE)
        self.vendor = patch.object(control, 'verify_vendor', return_value=True)
        for item in (self.sources, self.hardware, self.vendor):
            item.start(); self.addCleanup(item.stop)

    def test_exact_maps_seeds_budgets_and_backend_are_retained(self):
        for backend in ('jax', 'numba'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'primary';before=primary_fixture(root,backend)
                config,maps,fit=control.prepare_control(root)
                original=before['configuration.json']
                for key in ('seed','seed_batch','runs','workers','warmup','time_budget','thin',
                            'target','vi','beta','ess_enabled','wave_cpu_ids'):
                    self.assertEqual(config[key],original[key])
                self.assertEqual(config['algorithms'],control.METHODS)
                self.assertTrue(config['protocol'].endswith('-v9'))
                self.assertEqual(config['dcs_backend'],backend)
                self.assertEqual(config['step_tuning_policies'],{'SCS':'author_scs_fixed_01'})
                self.assertEqual(maps,before['maps.json']['maps'])
                self.assertEqual({r['seconds'] for r in fit.values()},{0.})
                self.assertEqual({r['original_fit_seconds'] for r in fit.values()},{1.25})
                self.assertTrue(all(r['map_reused_without_refitting'] for r in fit.values()))
                self.assertEqual(config['calibration_control_provenance']['presentation_source_differences'],{})
                for name,value in before.items():self.assertEqual(json.loads((root/name).read_text()),value)

    def test_only_renderer_source_difference_is_allowed_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'primary';files=primary_fixture(root)
            manifest=files['manifest.json'];manifest['source_hashes']['scripts/scs_100d_figures.py']='c'*64
            write(root/'manifest.json',manifest)
            config,_,_=control.prepare_control(root)
            self.assertEqual(config['calibration_control_provenance']['presentation_source_differences'],
                {'scripts/scs_100d_figures.py':{'primary':'c'*64,'current':'b'*64}})
            manifest['source_hashes']['dcs_bench/timed_radial.py']='d'*64
            write(root/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'scientific sources'):control.prepare_control(root)

    def test_incomplete_or_inconsistent_primary_is_rejected_before_running(self):
        changes = [('manifest.json', lambda x:x.update(status='running')),
            ('maps.json',lambda x:x['maps']['ball'].update(scale=99.)),
            ('method_maps.json',lambda x:x['SCS'].update(scale=99.)),
            ('summary.json',lambda x:x['cases']['SCS']['records'][0].update(seed=0)),
            ('summary.json',lambda x:x['cases']['SCS']['records'][0].update(tuning_policy='own_kernel_radial_pilot_v1'))]
        for name,change in changes:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'primary';files=primary_fixture(root);change(files[name]);write(root/name,files[name])
                with patch.object(runner,'run') as run, self.assertRaises(ValueError):
                    control.run(root,Path(directory)/'output')
                run.assert_not_called()

    def test_changed_cpu_allocation_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'primary';primary_fixture(root)
            with patch.object(runner,'cpu_info',return_value={'capacity':2,'cpu_ids':[0,1]}), \
                    self.assertRaisesRegex(ValueError,'CPU allocation'):
                control.prepare_control(root)

    def test_runner_reuses_maps_restores_prepare_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'primary';files=primary_fixture(root)
            output=Path(directory)/'control'
            original=runner.prepare
            def checked_run(config, path):
                maps,fit=runner.prepare(config)
                self.assertEqual(maps,files['maps.json']['maps'])
                self.assertTrue(all(v['seconds']==0 for v in fit.values()))
                self.assertEqual(path,output.resolve())
                raise RuntimeError('simulated sampling failure')
            with patch.object(runner,'run',side_effect=checked_run), \
                    patch.object(runner,'fit_map',side_effect=AssertionError('must not refit')), \
                    self.assertRaisesRegex(RuntimeError,'simulated'):
                control.run(root,output)
            self.assertIs(runner.prepare,original)
            with patch.object(control,'prepare_control') as prepare, self.assertRaises(FileExistsError):
                control.run(root,output)
            prepare.assert_not_called()


class FixedSCSKernelTests(unittest.TestCase):
    def test_fixed_warmup_matches_unmodified_author_states_and_uses_full_budget(self):
        import jax
        import numpy as np
        from dcs_bench import timed_radial
        mapping = SphereMap(5).parameters()
        config = dict(algorithm='SCS', dimension=5, nu=3., alpha={0:3.,1:-3.},
            warmup=13, thin=1, initial_scale=.1,
            step_tuning_policies={'SCS':'author_scs_fixed_01'})
        actual = timed_radial._Matched(mapping, config, 452)
        block, projection, initial, _ = timed_radial._components(mapping,
            timed_radial.scs_paper.target_spec(config), 'SCS')
        burn_key, _ = jax.random.split(jax.random.key(452))
        states, accepted = block(initial, burn_key, 13, .1)
        np.testing.assert_allclose(actual.state, states[-1], rtol=1e-12, atol=1e-12)
        self.assertEqual(actual.scale, .1)
        self.assertAlmostEqual(actual.warmup_acceptance, float(np.sum(accepted))/13)
        self.assertFalse(actual.tuning_metadata['step_size_adapted'])
        self.assertEqual(actual.tuning_metadata['fixed_step'], .1)
        _, production_key = jax.random.split(jax.random.key(452))
        _, batch_key = jax.random.split(production_key)
        expected_state, expected_prefixes, expected_saved = timed_radial._radial_batch(
            block, projection, states[-1], batch_key, .1, 1, 50.)
        prefixes, saved = actual.batch()
        np.testing.assert_allclose(actual.state, expected_state, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(prefixes, expected_prefixes, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(saved, expected_saved, rtol=1e-10, atol=1e-10)

    def test_fixed_policy_cannot_silently_change_step_or_apply_to_dcs(self):
        from dcs_bench.timed_radial import resolve_step_tuning
        for method, scale in [('DCS-BW',.1),('SCS',.2)]:
            with self.subTest(method=method), self.assertRaises(ValueError):
                resolve_step_tuning(dict(algorithm=method, warmup=10, initial_scale=scale,
                    step_tuning_policies={method:'author_scs_fixed_01'}))



if __name__=='__main__':unittest.main()
