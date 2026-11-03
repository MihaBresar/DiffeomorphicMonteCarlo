"""Full MH integration and separate experiment contract for chord reflection."""
import unittest
from unittest.mock import patch
import numpy as np
from dcs_bench import scs_paper, timed_radial as timed
from dcs_bench.ball_boundary import reflect_ball_proposal
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap
from scripts import reproduce_boundary_100d as boundary
from scripts import reproduce_scs_100d as original


class BoundaryComparisonTests(unittest.TestCase):
    def test_report_contains_all_three_methods_and_saturation_note(self):
        import csv
        import tempfile
        from pathlib import Path
        from tests.test_scs_100d_figures import example_summary
        from scripts.scs_100d_figures import render_report
        summary=example_summary()
        summary['config'].update(protocol='timed-boundary-dcs-scs-v1',mode='time',time_budget=1.,
                                 implementation_revision='jax-dcs-chord-reflection-v1',step_scale_ceiling=2.)
        summary['cases']['DCS-BW-reflect']=summary['cases'].pop('DCS-RWM')
        for name,case in summary['cases'].items():
            case['total_iterations']=3600
            for row in case['records']:
                row.update(iterations=1200,discarded_iterations=400,estimation_iterations=800,
                           scale_ceiling_reached=name=='DCS-BW-reflect')
                row.pop('checkpoints',None)
        with tempfile.TemporaryDirectory() as directory:
            render_report(Path(directory),summary)
            page=(Path(directory)/'paper_values.html').read_text()
            self.assertIn('boundary comparison',page)
            self.assertIn('step-size ceiling',page)
            self.assertIn('common upper step limit of 2.00',page)
            self.assertIn('DCS-BW-reflect',page)
            with (Path(directory)/'sample_counts.csv').open(newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([r['method'] for r in rows],list(boundary.METHODS))
            self.assertEqual([int(r['total_iterations']) for r in rows],[3600]*3)

    def test_configuration_matches_resources_and_preserves_original(self):
        a, b = original.configuration('paper'), boundary.configuration('paper')
        self.assertEqual(a['algorithms'], ['DCS-RWM', 'DCS-BW', 'SCS'])
        self.assertEqual(b['algorithms'], ['DCS-BW', 'DCS-BW-reflect', 'SCS'])
        self.assertEqual(b['protocol'], 'timed-boundary-dcs-scs-v2')
        self.assertEqual(b['step_scale_ceilings'], {'DCS-BW-reflect': 2.})
        self.assertNotIn('step_scale_ceiling', b)
        self.assertNotIn('step_scale_ceiling',a)
        for key in ('target','runs','time_budget','warmup','optimizer_steps','optimizer_samples',
                    'initial_scale','step_policy','burnin_fraction','seed','fit_seed'):
            self.assertEqual(a[key], b[key])

    def test_ceiling_resolution_preserves_defaults_and_historical_scalar_limits(self):
        for method in boundary.METHODS:
            with self.subTest(method=method):
                config = dict(boundary.configuration(), algorithm=method)
                expected = 2. if method == 'DCS-BW-reflect' else None
                self.assertEqual(timed.resolve_step_scale_ceiling(config), expected)
                historical = dict(config, step_scale_ceiling=3.)
                self.assertEqual(timed.resolve_step_scale_ceiling(historical),
                                 2. if method == 'DCS-BW-reflect' else 3.)
                historical.pop('step_scale_ceilings')
                self.assertEqual(timed.resolve_step_scale_ceiling(historical), 3.)
        config = dict(algorithm='DCS-BW', time_budget=.01, warmup=10, thin=10)
        for change in ({'step_scale_ceilings': []}, {'step_scale_ceilings': {'DCS-BW-typo': 2.}},
                       {'step_scale_ceilings': {'DCS-BW-reflect': np.inf}},
                       {'step_scale_ceilings': {'DCS-BW': .01}},
                       {'step_scale_ceilings': {'DCS-BW': True}},
                       {'step_scale_ceilings': {'DCS-BW': None}},
                       {'step_scale_ceiling': np.nan}, {'step_scale_ceiling': '2'},
                       {'initial_scale': 0.}):
            with self.subTest(change=change), patch.object(timed, '_backend') as backend:
                with self.assertRaises(ValueError):
                    timed.run_timed({}, {**config, **change}, 11)
                backend.assert_not_called()

    def test_v2_controls_reproduce_main_calibration_and_fixed_production_prefix(self):
        # Use real 100D kernels and nontrivial maps. Equal-time stopping is not
        # deterministic; compare a fixed prefix to isolate tuning and RNG use.
        scs_paper._author()
        import jax
        centre = np.linspace(-.05, .05, 100)
        shift = np.zeros(100)
        shift[:2] = [.1, -.05]
        mappings = {
            'DCS-BW': BallMap(100, mu=centre, scale=1.7, delta=shift).parameters(),
            'SCS': SphereMap(100, mu=centre, scale=1.7, observer=shift).parameters(),
        }
        main = original.configuration(overrides={'warmup': 128, 'thin': 10})
        revised = boundary.configuration(overrides={'warmup': 128, 'thin': 10})
        for method, mapping in mappings.items():
            for seed in (901001, 903001):
                with self.subTest(method=method, seed=seed):
                    before = timed._Matched(mapping, dict(main, algorithm=method), seed)
                    after = timed._Matched(mapping, dict(revised, algorithm=method), seed)
                    np.testing.assert_array_equal(before.state, after.state)
                    np.testing.assert_array_equal(jax.random.key_data(before.key), jax.random.key_data(after.key))
                    self.assertEqual(before.scale, after.scale)
                    self.assertEqual(before.warmup_acceptance, after.warmup_acceptance)
                    for _ in range(2):
                        old_prefixes, old_norms = before.batch()
                        new_prefixes, new_norms = after.batch()
                        np.testing.assert_array_equal(old_prefixes, new_prefixes)
                        np.testing.assert_array_equal(old_norms, new_norms)
                        np.testing.assert_array_equal(before.state, after.state)

    def test_records_use_resolved_method_ceiling(self):
        class Fake:
            compile_seconds, warmup_seconds, warmup_acceptance = 0., 0., .5
            scale, implementation = 2., 'test'
            def batch(self):
                return np.array([[0., 100., 50.], [0., 200., 100.], [0., 300., 150.]]), np.ones(30)
        configs = [
            (dict(algorithm='DCS-BW', step_scale_ceilings={'DCS-BW-reflect': 2.}), np.exp(15.), False),
            (dict(algorithm='DCS-BW-reflect', step_scale_ceilings={'DCS-BW-reflect': 2.}), 2., True),
            (dict(algorithm='DCS-BW', step_scale_ceiling=2.), 2., True),
            (dict(algorithm='DCS-BW', step_scale_ceiling=2., step_scale_ceilings={'DCS-BW': 3.}), 3., False),
        ]
        for limits, expected, reached in configs:
            config = dict(time_budget=.01, warmup=10, thin=10, **limits)
            with self.subTest(limits=limits), patch.object(timed, '_backend', return_value=Fake()), \
                    patch.object(timed.time, 'perf_counter', side_effect=[0., .02]), \
                    patch.object(timed.time, 'process_time', side_effect=[0., .01]):
                _, record = timed.run_timed({}, config, 11)
            self.assertEqual(record['step_scale_ceiling'], expected)
            self.assertEqual(record['scale_ceiling_reached'], reached)

    def test_mh_uses_reflected_endpoint_and_target_density_ratio(self):
        scs_paper._author()
        import jax
        import jax.numpy as jnp
        target = Target(dict(kind='skew_t',dimension=5,nu=3.,alpha={'0':3.}))
        mapping = BallMap(5,scale=1.7)
        block, _, _, _ = timed._components(mapping.parameters(), target.spec, 'DCS-BW-reflect')
        initial = jnp.array([.9,0.,0.,0.,0.])
        for seed in range(8):
            key,subkey=jax.random.split(jax.random.key(seed))
            proposed,valid=reflect_ball_proposal(initial, 2.*timed._uniform_ball(subkey,5))
            key,subkey=jax.random.split(key)
            ratio=mapping.log_density(target,np.asarray(proposed))-mapping.log_density(target,np.asarray(initial))
            accepted=bool(valid) and np.log(float(jax.random.uniform(subkey)))<ratio
            states,accepts=block(initial,jax.random.key(seed),1,2.)
            np.testing.assert_allclose(states[0], proposed if accepted else initial, atol=1e-13)
            self.assertEqual(float(accepts[0]),float(accepted))

    def test_uniform_transformed_target_accepts_and_saturates_tuning(self):
        scs_paper._author()
        import jax
        # This t_2 and beta=2 map have constant transformed density.
        target=Target(dict(kind='skew_t',dimension=5,nu=2.,alpha={}))
        mapping=BallMap(5,scale=np.sqrt(2.))
        block,_,initial,_=timed._components(mapping.parameters(),target.spec,'DCS-BW-reflect')
        states,accepted=jax.jit(lambda x,k:block(x,k,2000,3.))(initial,jax.random.key(71))
        self.assertTrue(np.all(np.asarray(accepted)==1.))
        self.assertTrue(np.all(np.linalg.norm(states,axis=1)<1.))
        _,scale,_=timed._calibrate(block,initial,jax.random.key(72),100,.1)
        self.assertAlmostEqual(float(scale)/np.exp(15.),1.,places=12)
        _,bounded,_=timed._calibrate(block,initial,jax.random.key(72),100,.1,2.)
        self.assertAlmostEqual(float(bounded),2.,places=12)


if __name__ == '__main__':
    unittest.main()
