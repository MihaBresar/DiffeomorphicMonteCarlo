"""Native billiard/RWM agreement with independent JAX geometry and physical MH."""
from copy import deepcopy
import unittest

import numpy as np

from dcs_bench.billiard import reflect_ball_billiard
from dcs_bench.numba_billiard import (_proposal_into, _transition_from_increment,
    _run_block, NumbaBilliard, BILLIARD_DOMAIN, RWM_DOMAIN)
from dcs_bench.numba_rwm import NumbaRWM
from dcs_bench import numba_radial as radial
from dcs_bench import timed_radial as timed
from dcs_bench.transforms import BallMap
from dcs_bench.targets import Target


class NativeBilliardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        jax.config.update('jax_enable_x64', True)

    def test_endpoints_match_jax_across_dimensions_and_scale_range(self):
        import jax
        rng = np.random.default_rng(591237)
        for dimension in (2, 5, 100, 500):
            count = 128
            draws = rng.normal(size=(count, dimension+2))
            points = draws[:, :dimension] / np.linalg.norm(draws, axis=1)[:, None]
            draws = rng.normal(size=(count, dimension+2))
            increments = draws[:, :dimension] / np.linalg.norm(draws, axis=1)[:, None]
            scales = np.exp(np.linspace(-30., 15., count))
            increments *= scales[:, None]
            expected, expected_valid = jax.jit(jax.vmap(reflect_ball_billiard))(points, increments)
            for i, (point, increment) in enumerate(zip(points, increments)):
                proposed = np.empty_like(point)
                valid = _proposal_into(point, increment, proposed)
                self.assertEqual(valid, bool(expected_valid[i]))
                self.assertTrue(valid)
                # Phase error grows with represented travel length; both
                # implementations use float64 reductions without fast-math.
                tolerance = max(2e-12, 2e-12*scales[i]*np.sqrt(dimension))
                np.testing.assert_allclose(proposed, expected[i], atol=tolerance, rtol=2e-12)
                self.assertLess(float(np.dot(proposed, proposed)), 1.)

    def test_interior_extension_diameter_grazing_and_invalid_proposals(self):
        cases = [(np.array([.999, 0.]), np.array([0., .02])),
                 (np.zeros(2), np.zeros(2)),
                 (np.array([.2, 0.]), np.array([7.4, 0.])),
                 (np.array([1.-1e-12, 0.]), np.array([0., np.exp(15.)]))]
        for start, step in cases:
            proposed = np.empty_like(start)
            valid = _proposal_into(start, step, proposed)
            expected, expected_valid = reflect_ball_billiard(start, step)
            self.assertEqual(valid, bool(expected_valid))
            self.assertTrue(valid)
            np.testing.assert_allclose(proposed, expected, atol=2e-12, rtol=2e-12)
        proposed = np.empty(2)
        self.assertTrue(_proposal_into(cases[0][0], cases[0][1], proposed))
        self.assertGreater(np.linalg.norm(proposed-sum(cases[0])), .01)
        for start, step in [(np.zeros(2), np.array([1., 0.])),
                            (np.ones(2), np.zeros(2)),
                            (np.array([np.nan, 0.]), np.ones(2)),
                            (np.zeros(2), np.array([np.inf, 0.])),
                            (np.zeros(2), np.array([np.nan, 0.])),
                            (np.zeros(2), np.array([1e300, 0.]))]:
            self.assertFalse(_proposal_into(start, step, np.empty(2)))

    def test_identical_increment_and_uniform_match_target_mh(self):
        import jax
        proposal = jax.jit(reflect_ball_billiard)
        rng = np.random.default_rng(919721)
        target = Target(dict(kind='skew_t', dimension=5, nu=3., alpha={'0': 3., '1': -3.}))
        for billiard in (False, True):
            mapping = BallMap(5, mu=[.2, -.1, 0., 0., 0.], scale=1.4,
                delta=np.zeros(5) if billiard else [.2, -.1, 0., 0., 0.])
            inputs = radial._inputs(mapping.parameters(), {'target': target.spec})
            for _ in range(80):
                draw = rng.normal(size=7)
                state = draw[:5]/np.linalg.norm(draw)
                increment = .8*rng.normal(size=5)
                logu = np.log(rng.random())
                if billiard:
                    proposed, valid = proposal(state, increment)
                    proposed = np.asarray(proposed)
                else:
                    proposed = state+increment
                    valid = np.dot(proposed, proposed) < 1.
                expected_accept = bool(valid and logu < mapping.log_density(target, proposed)-mapping.log_density(target, state))
                expected = proposed if expected_accept else state.copy()
                lp, radius = radial._evaluate(state, *inputs)
                next_lp, next_radius, accept = _transition_from_increment(state, lp, radius,
                    increment, logu, billiard, *inputs, np.empty(5))
                self.assertEqual(accept, expected_accept)
                np.testing.assert_allclose(state, expected, rtol=2e-12, atol=2e-12)
                self.assertAlmostEqual(next_lp, mapping.log_density(target, expected), places=10)
                # The physical forward map subtracts a rounded Mobius radius;
                # the native density uses the stable gap identity near the rim.
                np.testing.assert_allclose(next_radius, np.linalg.norm(mapping.forward(expected)[0]),
                    rtol=1e-10, atol=1e-11)

    def test_fused_statistics_use_all_states_and_gaussian_or_uniform_ball_generator(self):
        import jax
        proposal = jax.jit(reflect_ball_billiard)
        target = Target(dict(kind='skew_t', dimension=5, nu=3., alpha={'0': 3., '1': -3.}))
        for billiard in (False, True):
            mapping = BallMap(5, scale=1.6, mu=[.2, -.1, 0., 0., 0.],
                delta=np.zeros(5) if billiard else [.15, 0., 0., 0., 0.])
            inputs = radial._inputs(mapping.parameters(), {'target': target.spec})
            initial = np.array([.3, -.1, .2, 0., 0.]);scale=.8;seed=81721;threshold=2.
            state=initial.copy();rng=np.random.default_rng(seed);norms=[];accepted=[]
            for _ in range(100):
                draw = rng.standard_normal(7 if billiard else 5)
                increment = scale*(draw[:5]/np.linalg.norm(draw) if billiard else draw)
                logu=np.log(rng.random())
                if billiard:
                    proposed, valid=proposal(state,increment);proposed=np.asarray(proposed)
                else:
                    proposed=state+increment;valid=np.dot(proposed,proposed)<1.
                accept=bool(valid and logu < mapping.log_density(target,proposed)-mapping.log_density(target,state))
                if accept:state=proposed
                norms.append(np.linalg.norm(mapping.forward(state)[0]));accepted.append(accept)
            native_state=initial.copy();lp,radius=radial._evaluate(native_state,*inputs)
            _,_,prefix,saved=_run_block(np.random.default_rng(seed),native_state,lp,radius,
                scale,100,1,threshold,billiard,*inputs)
            np.testing.assert_allclose(native_state,state,atol=5e-11,rtol=5e-11)
            np.testing.assert_allclose(saved,norms,atol=5e-11,rtol=5e-11)
            expected=np.array([np.sum(np.asarray(norms)>=threshold),np.sum(norms),sum(accepted)])
            np.testing.assert_allclose(prefix[0],expected,rtol=5e-11,atol=5e-11)
            self.assertIn(False,accepted)
            self.assertTrue(np.any(np.diff(saved)==0.))

    def test_backends_reuse_calibration_and_freeze_step_without_mutating_maps(self):
        mapping=BallMap(5,scale=1.7,mu=[.2,-.1,0.,0.,0.],delta=[.2,-.1,0.,0.,0.]).parameters()
        original=deepcopy(mapping)
        for method,backend_type in [('DCS-RWM',NumbaRWM),('DCS-Billiard',NumbaBilliard)]:
            config=dict(algorithm=method,dimension=5,nu=3.,alpha={0:3.,1:-3.},thin=1,
                warmup=18,threshold=2.,time_budget=.001,dcs_backend='jax',coordinate_metrics=False)
            calibration=timed._Matched(mapping,config,77139)
            original_state=np.asarray(calibration.state).copy()
            backend=backend_type(mapping,{**config,'dcs_backend':'numba'},77139,calibration)
            np.testing.assert_array_equal(backend.state,original_state)
            self.assertEqual(backend.scale,calibration.scale)
            self.assertEqual(backend.warmup_seconds,calibration.warmup_seconds)
            self.assertEqual(backend.warmup_acceptance,calibration.warmup_acceptance)
            if method=='DCS-Billiard':
                self.assertEqual(backend.tuning_metadata['tuning_history'],calibration.tuning_metadata['tuning_history'])
                self.assertEqual(backend.tuning_metadata['effective_map']['delta'],[0.]*5)
                self.assertEqual(backend.tuning_metadata['effective_map']['mu'],mapping['mu'])
                self.assertEqual(backend.tuning_metadata['effective_map']['scale'],mapping['scale'])
            else:
                np.testing.assert_array_equal(backend.inputs[1],mapping['delta'])
            prefix,saved=backend.batch()
            self.assertEqual(prefix.shape,(3,3));self.assertEqual(saved.shape,(300,))
            self.assertTrue(np.all(np.isfinite(saved)))
            self.assertEqual(backend.scale,calibration.scale)
            np.testing.assert_array_equal(calibration.state,original_state)
            self.assertEqual(mapping,original)
        self.assertEqual(len({BILLIARD_DOMAIN,RWM_DOMAIN,radial.PRODUCTION_DOMAIN}),3)

    def test_radial_recording_matches_optional_coordinate_recording(self):
        mapping=BallMap(5,scale=1.7,delta=[.15,0.,0.,0.,0.]).parameters()
        for method,backend_type in [('DCS-RWM',NumbaRWM),('DCS-Billiard',NumbaBilliard)]:
            config=dict(algorithm=method,dimension=5,nu=3.,alpha={0:3.,1:-3.},thin=1,
                warmup=18,threshold=2.,time_budget=.001,dcs_backend='jax',coordinate_metrics=False)
            calibration=timed._Matched(mapping,config,89135)
            radial_backend=backend_type(mapping,config,89135,calibration)
            extended_backend=backend_type(mapping,{**config,'coordinate_metrics':True},89135,calibration)
            radial_prefix,radial_saved=radial_backend.batch()
            extended_prefix,extended_saved=extended_backend.batch()
            np.testing.assert_array_equal(radial_backend.state,extended_backend.state)
            np.testing.assert_allclose(radial_prefix,extended_prefix[:,:3],rtol=1e-14)
            np.testing.assert_array_equal(radial_saved,extended_saved[:,0])


if __name__=='__main__':unittest.main()
