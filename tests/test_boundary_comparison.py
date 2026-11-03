"""Full MH integration and separate experiment contract for chord reflection."""
import unittest
import numpy as np
from dcs_bench import scs_paper, timed_radial as timed
from dcs_bench.ball_boundary import reflect_ball_proposal
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap
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
        self.assertEqual(b['protocol'], 'timed-boundary-dcs-scs-v1')
        self.assertEqual(b['step_scale_ceiling'],2.)
        self.assertNotIn('step_scale_ceiling',a)
        for key in ('target','runs','time_budget','warmup','optimizer_steps','optimizer_samples',
                    'initial_scale','step_policy','burnin_fraction','seed','fit_seed'):
            self.assertEqual(a[key], b[key])

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
