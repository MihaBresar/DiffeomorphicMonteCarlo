"""Reflected Gaussian mixture law, independent MH checks and pilot accounting."""
import unittest

import numpy as np
from scipy.special import gammaln, logsumexp

from dcs_bench import rwm_rotation
from dcs_bench.references import skew_t_iid
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap
from tests.test_numba_asymmetry import direct_log_prob
from tests.test_numba_radial import reflected_reference


def target_spec(d, nu=3., alpha=None):
    return dict(kind='skew_t', dimension=d, nu=nu,
                alpha={'0': 3., '1': -3.} if alpha is None else alpha)


def fitted_map(d, fitted):
    mu, delta = np.zeros(d), np.zeros(d)
    if fitted:
        mu[:2], delta[:2] = [.2, -.1], [.15, -.1]
    return BallMap(d, mu=mu, delta=delta, scale=1.3 if fitted else 1.)


class RwmRotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        jax.config.update('jax_enable_x64', True)

    def test_initial_scale_matches_gaussian_and_ball_second_moments(self):
        for d in (2, 6, 100, 500):
            sigma = rwm_rotation.initial_sigma(d, .1)
            self.assertAlmostEqual(d*sigma*sigma, .1**2*d/(d+2), places=15)
        for d, radius in ((1,.1),(2.5,.1),(True,.1),(6,True),(6,0.),(6,np.nan),(6,np.inf)):
            with self.subTest(d=d, radius=radius), self.assertRaises(ValueError):
                rwm_rotation.initial_sigma(d, radius)

    def test_reflected_signed_chi_density_is_symmetric(self):
        # Independent image-sum formula on a chord. Gaussian vector lengths
        # follow chi_d; choose a sign with probability1/2 on an unoriented line.
        def log_signed_chi(t, d, sigma):
            r = np.abs(t)/sigma
            with np.errstate(divide='ignore'):
                return ((d-1)*np.log(r)-.5*r*r-d/2*np.log(2.)
                        -gammaln(d/2)-np.log(sigma))
        def image_sum(x, y, length, d, sigma):
            periods = 2*np.arange(-500,501)*length
            return logsumexp(np.r_[log_signed_chi(y-x+periods,d,sigma),
                                     log_signed_chi(y+x+periods,d,sigma)])
        for d in (6,100,500):
            for sigma in (.02,.2,2.):
                with self.subTest(d=d, sigma=sigma):
                    self.assertAlmostEqual(image_sum(.17,.91,1.3,d,sigma),
                                           image_sum(.91,.17,1.3,d,sigma), places=11)

    def test_complete_path_matches_independent_physical_coordinate_mh(self):
        import jax
        import jax.numpy as jnp
        # The oracle rotates physical x-mu and uses pi(x')/pi(x), avoiding
        # the implementation's Mobius proposal Jacobian formula completely.
        d, count, scales = 6, 120, np.array([.16,.7])
        spec = target_spec(d)
        for fitted in (False,True):
            mapping = fitted_map(d,fitted)
            parts = rwm_rotation.build_parts(mapping.parameters(),spec)
            initial = mapping.inverse(np.ones(d))
            key = jax.random.key(57931)
            state, rows, accepted_rows, choices = initial.copy(), [], [], []
            for _ in range(count):
                key, selection_key, normal_key, accept_key = jax.random.split(key,4)
                normal = np.asarray(jax.random.normal(normal_key,(d,)))
                logu = float(jnp.log(jax.random.uniform(accept_key)))
                angular = bool(jax.random.bernoulli(selection_key,.5))
                physical, old_jac = mapping.forward(state)
                if angular:
                    centered = physical-mapping.mu
                    radius = np.linalg.norm(centered)
                    direction = centered/radius
                    tangent = normal-direction*np.dot(direction,normal)
                    tangent /= np.linalg.norm(tangent)
                    theta = 2*np.arctan(scales[1])
                    proposed_physical = mapping.mu+np.cos(theta)*centered+np.sin(theta)*radius*tangent
                    proposed = mapping.inverse(proposed_physical)
                    ratio = direct_log_prob(spec,proposed_physical)-direct_log_prob(spec,physical)
                else:
                    proposed = reflected_reference(state,scales[0]*normal)
                    proposed_physical, new_jac = mapping.forward(proposed)
                    ratio = (direct_log_prob(spec,proposed_physical)+new_jac
                             -direct_log_prob(spec,physical)-old_jac)
                accepted = logu < ratio
                if accepted:
                    state = proposed
                rows.append(state.copy());accepted_rows.append(accepted);choices.append(angular)
            states, accepts = jax.jit(lambda key: parts['block'](
                jnp.asarray(initial),key,count,jnp.asarray(scales)))(jax.random.key(57931))
            with self.subTest(vi=fitted):
                np.testing.assert_array_equal(accepts,accepted_rows)
                np.testing.assert_allclose(states,rows,rtol=1e-9,atol=1e-10)
                self.assertGreater(sum(choices),30)
                self.assertLess(sum(choices),90)

    def test_skew_target_invariance_from_exact_starts_with_and_without_vi(self):
        import jax
        import jax.numpy as jnp
        rng = np.random.default_rng(678240)
        for d in (6,100):
            spec = target_spec(d)
            physical = skew_t_iid(Target(spec),rng,6000)
            for fitted in (False,True):
                mapping = fitted_map(d,fitted)
                parts = rwm_rotation.build_parts(mapping.parameters(),spec)
                keys = jax.random.split(jax.random.key(98910+d),len(physical))
                scales = jnp.array([4*rwm_rotation.initial_sigma(d),2.])
                def step(z,key):
                    states,_ = parts['block'](z,key,1,scales)
                    return parts['projection'](states)[0]
                outputs = np.asarray(jax.jit(jax.vmap(step))(jnp.asarray(mapping.inverse(physical)),keys))
                def features(x):
                    return np.column_stack((np.tanh(x[:,:3]),np.tanh(np.log(np.linalg.norm(x,axis=1))),
                                             x[:,0]>3,x[:,1]<-3,x[:,2]>3))
                delta = features(outputs)-features(physical)
                se = delta.std(axis=0,ddof=1)/np.sqrt(len(delta))
                with self.subTest(d=d,vi=fitted):
                    np.testing.assert_array_less(np.abs(delta.mean(axis=0)),6*se+1e-12)

    def test_uniform_ball_accepts_all_preserves_moments_and_selects_one_component(self):
        import jax
        import jax.numpy as jnp
        d, n = 7, 6000
        parts = rwm_rotation.build_parts(BallMap(d,scale=np.sqrt(2.)).parameters(),
                                         target_spec(d,nu=2.,alpha={}))
        rng = np.random.default_rng(872961)
        starts = rng.normal(size=(n,d))
        starts /= np.linalg.norm(starts,axis=1,keepdims=True)
        starts *= rng.random((n,1))**(1/d)
        keys = jax.random.split(jax.random.key(3916),n)
        def step(z,key):
            states,accepted = parts['block'](z,key,1,jnp.array([.4,1.]))
            return states[0],accepted[0]
        output,accepted = jax.jit(jax.vmap(step))(jnp.asarray(starts),keys)
        np.testing.assert_array_equal(accepted,1.)
        output = np.asarray(output)
        self.assertEqual(output.shape,(n,d))
        delta = output**2-starts**2
        se = delta.std(axis=0,ddof=1)/np.sqrt(n)
        np.testing.assert_array_less(np.abs(delta.mean(axis=0)),6*se+1e-12)
        angular_fraction = np.mean(np.abs(np.linalg.norm(output,axis=1)-np.linalg.norm(starts,axis=1))<1e-12)
        self.assertGreater(angular_fraction,.46)
        self.assertLess(angular_fraction,.54)

    def test_real_split_calibration_freezes_two_selected_scales_and_exact_budget(self):
        import jax
        mapping = fitted_map(6,True)
        parts = rwm_rotation.build_parts(mapping.parameters(),target_spec(6))
        for warmup in (7,60):
            result = jax.jit(lambda key: rwm_rotation.calibrate(parts,key,warmup))(
                jax.random.key(6289))
            state,scales,accepts,histories = result
            with self.subTest(warmup=warmup):
                self.assertTrue(np.all(np.isfinite(state)))
                self.assertTrue(np.all(np.asarray(scales)>0))
                self.assertLessEqual(float(accepts),warmup)
                for index,(history,count) in enumerate(zip(histories,(warmup//2,warmup-warmup//2))):
                    history = np.asarray(history)
                    self.assertEqual(int(history[:,:,1].sum()),count)
                    self.assertTrue(np.all(history[:,:,4].sum(axis=1)==1))
                    winner = np.argmax(history[-1,:,4])
                    self.assertAlmostEqual(float(scales[index]),float(history[-1,winner,0]),places=12)
                # Repeated fixed-state/key batches must not alter either scale.
                before = np.asarray(scales).copy()
                first = parts['block'](state,jax.random.key(912),3,scales)
                second = parts['block'](state,jax.random.key(912),3,scales)
                np.testing.assert_array_equal(first[0],second[0])
                np.testing.assert_array_equal(scales,before)


if __name__ == '__main__':
    unittest.main()
