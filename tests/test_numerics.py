"""Scientific regression checks independent of the benchmark's summary code."""
import run_experiment  # CPU and float64 policy, before importing JAX.
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from scipy import integrate, linalg, special, stats
import jax
import jax.numpy as jnp
from dcs_bench.targets import Target, ROOT
from dcs_bench.transforms import BallMap, SphereMap
from dcs_bench.references import skew_t_iid, radial_truth, mcmc_diagnostics, posterior_reference
from dcs_bench.samplers import MetropolisKernel
from dcs_bench.tuning import uniform_ball, uniform_cap
from dcs_bench.plotting import mse_summary
from dcs_bench.engine import load_config, save_chain, read_chain, experiment_names
from dcs_bench.api import sample
jax.config.update('jax_enable_x64', True)


class ScientificTests(unittest.TestCase):
    def test_public_api_returns_fixed_kernel_draws(self):
        draws, info = sample(lambda x: -2.5*np.log1p(x@x/3), np.zeros(2), draws=200, warmup=100, seed=18)
        self.assertEqual(draws.shape, (200, 2))
        self.assertFalse(info['adaptation_during_sampling'])
        self.assertEqual(info['iterations'], 200)
        with self.assertRaises(ValueError):
            load_config('skew-t-100d-fixed', 'paper', {'time_budget': 10})

    def test_ball_roundtrip_and_jacobian(self):
        z = np.array([.23, -.31, .18])
        for beta in [.2, 1., 2.]:
            mapping = BallMap(3, beta, mu=[1, -2, 3], scale=1.7, delta=[.12, -.15, .05])
            x, logjac = mapping.forward(z)
            assert_allclose(mapping.inverse(x), z, atol=2e-14)
            h = 1e-6
            jac = np.column_stack([(mapping.forward(z+h*e)[0]-mapping.forward(z-h*e)[0])/(2*h) for e in np.eye(3)])
            assert_allclose(np.linalg.slogdet(jac)[1], logjac, atol=1e-8)

    def test_sphere_roundtrip_and_surface_jacobian(self):
        mapping = SphereMap(3, mu=[1, -2, 3], scale=1.7, observer=[.2, -.1, .1])
        z = np.array([.2, -.3, .1, -np.sqrt(.86)])
        x, logjac = mapping.forward(z)
        assert_allclose(mapping.inverse(x), z, atol=2e-14)
        tangent = linalg.null_space(z[None, :])
        h = 1e-6
        def move(v):
            q = z+v
            return mapping.forward(q/np.linalg.norm(q))[0]
        jac = np.column_stack([(move(h*e)-move(-h*e))/(2*h) for e in tangent.T])
        assert_allclose(np.linalg.slogdet(jac)[1], logjac, atol=1e-8)

    def test_stable_ball_density_and_no_artificial_cutoff(self):
        target = Target(dict(kind='skew_t', dimension=4, nu=2, alpha={}))
        mapping = BallMap(4, beta=2, scale=np.sqrt(2))
        origin = mapping.log_density(target, np.zeros(4))
        for radius in [.2, .9, .99999999, 1-1e-12]:
            z = np.array([radius, 0, 0, 0])
            self.assertTrue(np.isfinite(mapping.log_density(target, z)))
            # This t target transforms exactly to a uniform ball density.
            assert_allclose(mapping.log_density(target, z), origin, atol=1e-12)
        self.assertEqual(mapping.log_density(target, np.array([1., 0, 0, 0])), -np.inf)
        skew = Target(dict(kind='skew_t', dimension=4, nu=3, alpha={'0': 20, '1': -30}))
        mapping = BallMap(4, 2, mu=[.5, -.4, .1, .2], scale=1.2, delta=[.1, 0, 0, 0])
        for z in [np.zeros(4), np.array([-.3, .6, 0, 0]), np.array([.9999, 0, 0, 0])]:
            x, lj = mapping.forward(z)
            assert_allclose(mapping.log_density(skew, z), skew.log_prob(x)+lj, atol=2e-9)

    def test_numpy_jax_targets_agree(self):
        for name in experiment_names():
            for spec in load_config(name, 'paper')['targets']:
                target = Target(spec)
                x = target.initial()+np.linspace(-.15, .2, target.d)
                with self.subTest(name=name, spec=spec):
                    assert_allclose(target.log_prob(x), target.log_prob(jnp.asarray(x), 'jax'), rtol=2e-12, atol=2e-10)
                    grad = jax.grad(lambda q: target.log_prob(q, 'jax'))(jnp.asarray(x))
                    self.assertTrue(np.all(np.isfinite(grad)))

    def test_garch_matches_independent_flat_physical_density(self):
        target = Target(load_config('garch11', 'paper')['targets'][0])
        for x in [np.array([5., -.7, -.9, .8]), np.array([4.7, -2., 2., -1.])]:
            mu, omega, alpha, beta = target.physical(x)
            variance, loglik = target.sigma1**2, 0.
            for t, y in enumerate(target.y):
                if t:
                    variance = omega+alpha*(target.y[t-1]-mu)**2+beta*variance
                loglik += stats.norm.logpdf(y, loc=mu, scale=np.sqrt(variance))
            jac = jax.jacfwd(lambda q: target.physical(q, 'jax'))(jnp.asarray(x))
            expected = loglik+np.linalg.slogdet(jac)[1]+len(target.y)/2*np.log(2*np.pi)
            assert_allclose(target.log_prob(x), expected, atol=1e-10)
            # Conditional Uniform(0,1-alpha) WITHOUT the correction is wrong.
            self.assertGreater(abs(expected-(expected-np.log1p(-alpha))), .1)

    def test_eight_schools_parameterizations_have_same_physical_target(self):
        centered = Target(dict(kind='eight_schools', dimension=10, parameterization='centered'))
        noncentered = Target(dict(kind='eight_schools', dimension=10, parameterization='noncentered'))
        z = np.r_[np.linspace(-1, 1, 8), 3., np.log(2.)]
        x = np.r_[3+2*z[:8], z[-2:]]
        assert_allclose(centered.physical(x), noncentered.physical(z))
        assert_allclose(noncentered.log_prob(z), centered.log_prob(x)+8*np.log(2), atol=1e-12)

    def test_normal_cdf_skew_t_reference(self):
        target = Target(dict(kind='skew_t', dimension=1, nu=3, alpha={'0': 2.}))
        rng = np.random.default_rng(182)
        draws = skew_t_iid(target, rng, 100000)[:, 0]
        truth, _ = integrate.quad(lambda x: np.exp(target.log_prob(np.array([x]))), -np.inf, 0, epsabs=1e-10)
        self.assertLess(abs(np.mean(draws < 0)-truth), .003)
        self.assertLess(abs(np.mean(draws*draws <= stats.f.ppf(.8, 1, 3))-.8), .005)

    def test_radial_exact_values(self):
        target = Target(dict(kind='skew_t', dimension=200, nu=3, alpha={'0': 20, '1': -30}))
        _, value, _ = radial_truth(target, 50)
        assert_allclose(value, [.029191445495738842, 19.519685714651676], atol=1e-11)
        _, heavy, _ = radial_truth(Target(dict(kind='skew_t', dimension=50, nu=.5, alpha={})), 50)
        self.assertTrue(np.all(np.isfinite(heavy)))

    def test_uniform_stationarity_of_ball_kernels(self):
        target = Target(dict(kind='skew_t', dimension=3, nu=2, alpha={}))
        mapping = BallMap(3, beta=2, scale=np.sqrt(2))
        for algorithm in ['DCS-BW', 'DCS-RWM', 'DCS-HnR']:
            kernel = MetropolisKernel(target, algorithm, mapping, seed=12)
            rng = np.random.default_rng(83)
            starts = uniform_ball(rng, 6000, 3)
            final = []
            for z in starts:
                kernel.state, kernel.lp = z, mapping.log_density(target, z)
                kernel.step(.35)
                final.append(kernel.state.copy())
            final = np.asarray(final)
            self.assertLess(abs(np.mean(np.sum(final*final, axis=1))-.6), .018)
            self.assertLess(np.max(abs(final.mean(axis=0))), .025)

    def test_scs_uniform_cap_stationarity(self):
        class UniformTarget:
            d = 3
            def initial(self): return np.zeros(3)
            def log_prob(self, x): return 0.
        class IdentityCap:
            latitude = 1.1
            def inverse(self, x): return np.array([0., 0., 0., -1.])
            def log_density(self, target, z): return 0. if z[-1] < self.latitude-1 else -np.inf
        kernel = MetropolisKernel(UniformTarget(), 'SCS', IdentityCap(), seed=39)
        starts = uniform_cap(np.random.default_rng(29), 10000, 3, 1.1)
        final = []
        for z in starts:
            kernel.state, kernel.lp = z, 0.
            kernel.step(.8)
            final.append(kernel.state.copy())
        final = np.asarray(final)
        self.assertTrue(np.all(final[:, -1] < .1))
        self.assertLess(np.max(abs(final.mean(axis=0)-starts.mean(axis=0))), .025)
        self.assertLess(stats.ks_2samp(starts[:, -1], final[:, -1]).statistic, .03)

    def test_rmse_averages_chain_squared_errors(self):
        result = mse_summary(np.array([[0., 0.], [2., 4.]]), np.array([1., 2.]))
        assert_allclose(result['rmse'], [1., 2.])
        # A pooled estimator would erroneously have RMSE zero here.

    def test_diagnostics_do_not_turn_failure_into_zero_ess(self):
        diagnostic = mcmc_diagnostics(np.zeros((2, 20, 1)), ['tail'])
        self.assertIsNone(diagnostic['ess_bulk']['tail'])
        self.assertIsNone(diagnostic['mcse_mean']['tail'])

    def test_posterior_reference_order(self):
        for name in ['garch11', 'eight-schools']:
            target = Target(load_config(name, 'paper')['targets'][0])
            draws, meta = posterior_reference(target)
            self.assertEqual(draws.shape, (10, 1000, target.d))
            self.assertFalse(meta['exact'])

    def test_paper_settings_and_checkpoint_integrity(self):
        self.assertEqual(load_config('skew-t-200d-baselines', 'paper')['targets'][0]['dimension'], 200)
        self.assertEqual(load_config('skew-t-100d-fixed', 'paper')['targets'][0]['nu'], 3)
        self.assertEqual(load_config('skew-t-50d-nu-0.5', 'paper')['targets'][0]['alpha'], {})
        self.assertEqual(load_config('logistic-regression', 'paper')['time_budget'], 30)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'chain.npz'
            save_chain(path, np.ones((10, 3)), {'fingerprint': 'abc'})
            self.assertEqual(read_chain(path, 'abc')[0].shape, (10, 3))
            with self.assertRaises(ValueError): read_chain(path, 'different')
            save_chain(path, np.empty((0, 3)), {'fingerprint': 'abc'})
            with self.assertRaises(ValueError): read_chain(path, 'abc')


if __name__ == '__main__':
    unittest.main()
