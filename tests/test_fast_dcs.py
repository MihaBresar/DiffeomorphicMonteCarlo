"""Independent NumPy/SciPy oracles for the compiled DCS loop."""
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import log_ndtr

from dcs_bench.fast_dcs import _adapt, _evaluate, _inputs, _log_ndtr, _transition, run_chain
from dcs_bench.samplers import MetropolisKernel
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap
from dcs_bench.tuning import DualAveraging


class FastDCSTests(unittest.TestCase):
    def test_scipy_c_api_is_the_exact_real_log_cdf(self):
        for value in [-1e6, -100., -40., -8., -1., 0., 1., 8., 40., 1e6]:
            self.assertEqual(_log_ndtr(value, 0), log_ndtr(value))

    def test_compiled_density_and_cached_radius_match_canonical_map(self):
        rng = np.random.default_rng(182)
        for d in (3, 100):
            target = Target({'kind': 'skew_t', 'dimension': d, 'nu': 3.,
                             'alpha': {'0': 100., '1': -100.}})
            for beta in (.2, 2., 3.):
                mapping = BallMap(d, beta=beta, mu=rng.normal(size=d)*.1,
                                  delta=np.r_[.2, np.zeros(d-1)], scale=1.7)
                _, mu, delta, alpha, constants = _inputs(mapping, target.spec)
                points = rng.normal(size=(20, d))
                points *= rng.uniform(.01, .999, (20, 1))/np.linalg.norm(points, axis=1)[:, None]
                for z in points:
                    lp, radius = _evaluate(z, mu, delta, alpha, constants)
                    x, jac = mapping.forward(z)
                    assert_allclose(lp, target.log_prob(x)+jac, rtol=2e-12, atol=5e-10)
                    assert_allclose(radius, np.linalg.norm(x), rtol=3e-12, atol=2e-12)
                self.assertEqual(_evaluate(np.ones(d), mu, delta, alpha, constants)[0], -np.inf)

    def test_transitions_preserve_numpy_rng_proposals_and_acceptance(self):
        d = 100
        target = Target({'kind': 'skew_t', 'dimension': d, 'nu': 3.,
                         'alpha': {'0': 100., '1': -100.}})
        mapping = BallMap(d, scale=1.2, mu=np.r_[.3, -.3, np.zeros(d-2)],
                          delta=np.r_[.1, -.1, np.zeros(d-2)])
        _, mu, delta, alpha, constants = _inputs(mapping, target.spec)
        for algorithm in ('DCS-RWM', 'DCS-BW'):
            oracle = MetropolisKernel(target, algorithm, mapping, 3801)
            rng = np.random.default_rng(3801)
            z = oracle.state.copy()
            lp, radius = _evaluate(z, mu, delta, alpha, constants)
            scale = .025 if algorithm == 'DCS-RWM' else .25
            for _ in range(1000):
                z, lp, radius, probability, accepted = _transition(
                    rng, z, lp, radius, scale, algorithm == 'DCS-BW', mu, delta, alpha, constants)
                expected_probability, expected_acceptance = oracle.step(scale)
                self.assertEqual(accepted, expected_acceptance)
                assert_allclose(probability, expected_probability, rtol=1e-10, atol=2e-12)
                assert_allclose(z, oracle.state, rtol=2e-12, atol=2e-13)
                assert_allclose(radius, np.linalg.norm(oracle.draw()), rtol=2e-11)
            # All random draws, including rejection uniforms, are consumed in
            # the same order, without replacing the generator algorithm.
            self.assertEqual(rng.random(), oracle.rng.random())

    def test_adaptation_and_all_iteration_summaries_match_reference(self):
        d = 5
        target = Target({'kind': 'skew_t', 'dimension': d, 'nu': 3., 'alpha': {'0': 2.}})
        mapping = BallMap(d, scale=1.3, delta=[.1, 0., 0., 0., 0.])
        for algorithm in ('DCS-RWM', 'DCS-BW'):
            config = {'target': target.spec, 'algorithm': algorithm, 'warmup': 100,
                      'iterations': 210, 'thin': 7, 'threshold': 4.}
            called = []
            traces, record = run_chain(mapping.parameters(), config, 882, lambda: called.append(True))
            self.assertEqual(called, [True])
            oracle = MetropolisKernel(target, algorithm, mapping, 882)
            adapter, scale = DualAveraging(.1), .1
            summaries, accepted = [], 0
            for i in range(config['warmup']+config['iterations']):
                probability, did_accept = oracle.step(scale)
                adapter.update(probability)
                scale = adapter.final()
                if i >= config['warmup']:
                    radius = np.linalg.norm(oracle.draw())
                    summaries.append([float(radius >= config['threshold']), radius])
                    accepted += did_accept
            summaries = np.asarray(summaries)
            assert_allclose(traces, summaries[6::7], rtol=2e-9, atol=2e-9)
            assert_allclose(record['mean_all_iterations'], summaries.mean(axis=0), rtol=2e-10)
            self.assertEqual(record['acceptance_rate'], accepted/config['iterations'])
            assert_allclose(record['final_scale'], scale, rtol=2e-10)
            for point in record['checkpoints']:
                assert_allclose(point['mean_all_iterations'], summaries[:point['iterations']].mean(axis=0), rtol=2e-10)

    def test_smoothed_adaptation_matches_reference_for_fixed_probabilities(self):
        adapter, hbar, logbar = DualAveraging(.1), 0., 0.
        for i, probability in enumerate(np.random.default_rng(628).beta(2., 6., size=5000), 1):
            scale, hbar, logbar = _adapt(probability, i, hbar, logbar)
            adapter.update(probability)
            # The supplied first-version algorithm has no scale cap; the
            # modern generic adapter.final() clips its returned log scale.
            assert_allclose(scale, np.exp(adapter.logbar), rtol=3e-13)
            assert_allclose([hbar, logbar], [adapter.hbar, adapter.logbar], rtol=3e-13, atol=2e-14)

    def test_thinning_does_not_change_all_iteration_estimates(self):
        mapping = BallMap(100).parameters()
        config = {'algorithm': 'DCS-BW', 'warmup': 100, 'iterations': 1000, 'thin': 1}
        dense, a = run_chain(mapping, config, 781)
        sparse, b = run_chain(mapping, dict(config, thin=17), 781)
        np.testing.assert_array_equal(sparse, dense[16::17])
        self.assertEqual(a['mean_all_iterations'], b['mean_all_iterations'])
        self.assertEqual(a['final_scale'], b['final_scale'])


if __name__ == '__main__':
    unittest.main()
