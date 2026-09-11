"""Exact first-third exclusion, frozen kernels and equivalent batch statistics."""
import unittest
from unittest.mock import patch

import numpy as np

from dcs_bench import timed_radial as timed
from dcs_bench import fast_dcs, scs_paper
from dcs_bench.transforms import BallMap, SphereMap
from dcs_bench.targets import Target


def prefixes_from(norms, accepted, threshold=50.):
    values = np.column_stack((norms >= threshold, norms, accepted))
    return np.cumsum(values, axis=0)[99::100]


class TimedRadialTests(unittest.TestCase):
    def test_exact_third_burn_falls_inside_batches_and_retains_rejections(self):
        for batches in (1, 2, 4, 7):
            # Repeated radii represent states repeated following rejection.
            norms = np.repeat(np.arange(1., batches * 150 + 1), 2)
            accepted = np.tile([1., 0.], batches * 150)
            prefixes = prefixes_from(norms, accepted)
            for thin in (1, 10, 50):
                with self.subTest(batches=batches, thin=thin):
                    trace, count, burn, used, total, sums = timed._postburn(
                        prefixes, norms[thin-1::thin], thin, 50.)
                    reference = norms[len(norms)//3:]
                    self.assertEqual(count, len(norms))
                    self.assertEqual(burn, len(norms)//3)
                    self.assertEqual(used, len(reference))
                    self.assertEqual(sums[0], np.sum(reference >= 50))
                    self.assertEqual(sums[1], np.sum(reference))
                    self.assertEqual(sums[2], np.sum(accepted[burn:]))
                    np.testing.assert_array_equal(trace[:, 1], reference[thin-1::thin])
                    self.assertEqual(total[1], norms.sum())

    def test_equal_batch_clock_overshoot_and_exact_postburn_record(self):
        norms = np.repeat(np.arange(1., 301.), 2)
        accepted = np.tile([1., 0.], 300)
        class Fake:
            compile_seconds, warmup_seconds, warmup_acceptance = 4., 3., .2
            scale, implementation, index = .125, 'test', 0
            def batch(self):
                selected = slice(self.index*300, (self.index+1)*300)
                self.index += 1
                values = norms[selected]
                return prefixes_from(values, accepted[selected]), values[9::10]
        calls = []
        config = dict(algorithm='DCS-BW', time_budget=.015, warmup=100, thin=10)
        with patch.object(timed, '_backend', return_value=Fake()), \
                patch.object(timed.time, 'perf_counter', side_effect=[0., .01, .02]), \
                patch.object(timed.time, 'process_time', side_effect=[8., 8.01]):
            trace, result = timed.run_timed({}, config, 8,
                before_sampling=lambda: calls.append('ready'), after_sampling=lambda: calls.append('finished'))
        self.assertEqual(calls, ['ready', 'finished'])
        self.assertEqual(result['iterations'], 600)
        self.assertEqual(result['discarded_iterations'], 200)
        self.assertEqual(result['estimation_iterations'], 400)
        self.assertEqual(result['retained_draws'], 40)
        np.testing.assert_allclose(result['mean_all_iterations'],
            [np.mean(norms[200:] >= 50), np.mean(norms[200:])])
        self.assertEqual(result['tail_count'], np.sum(norms[200:] >= 50))
        self.assertEqual(result['norm_sum'], norms[200:].sum())
        self.assertAlmostEqual(result['budget_overshoot_seconds'], .005)
        self.assertEqual(result['warmup_seconds'], 3.)
        self.assertEqual(result['compile_seconds'], 4.)
        self.assertEqual(result['frozen_scale'], .125)
        self.assertFalse(result['adaptation_during_sampling'])
        self.assertEqual(result['acceptance'], .5)
        self.assertEqual(result['postburn_acceptance'], .5)

    def test_frozen_dcs_batch_matches_original_transition_without_adapting(self):
        for algorithm in ('DCS-RWM', 'DCS-BW'):
            with self.subTest(algorithm=algorithm):
                config = dict(algorithm=algorithm, dimension=5, nu=3.,
                              alpha={0: 3., 1: -3.}, thin=10, warmup=37)
                mapping = BallMap(5, beta=2.)
                backend = timed._DCS(mapping.parameters(), config, 774)
                _, mu, delta, alpha, constants = fast_dcs._inputs(mapping, config)
                z = mapping.inverse(np.ones(5))
                lp, radius = fast_dcs._evaluate(z, mu, delta, alpha, constants)
                rng = np.random.default_rng(774)
                warm = fast_dcs._run_block(rng, z, lp, radius, .1, 0., 0., 0, 37,
                    38, 50., algorithm == 'DCS-BW', mu, delta, alpha, constants)
                z, lp, radius, frozen = warm[:4]
                reference, accepted = [], []
                for _ in range(300):
                    z, lp, radius, _, accept = fast_dcs._transition(rng, z, lp, radius,
                        frozen, algorithm == 'DCS-BW', mu, delta, alpha, constants)
                    reference.append(radius)
                    accepted.append(accept)
                prefix, saved = backend.batch()
                np.testing.assert_allclose(prefix, prefixes_from(np.array(reference), accepted), rtol=1e-12)
                np.testing.assert_allclose(saved, np.array(reference)[9::10], rtol=1e-12)
                np.testing.assert_allclose(backend.state, z, rtol=1e-12, atol=1e-12)
                self.assertEqual(backend.scale, frozen)

    def test_scs_batch_is_original_author_scan_with_equivalent_summaries(self):
        core = scs_paper._author()
        import jax
        import jax.numpy as jnp
        config = dict(algorithm='SCS', dimension=5, nu=3., alpha={0: 3., 1: -3.},
                      thin=10, warmup=17)
        mapping = SphereMap(5).parameters()
        backend = timed._SCS(mapping, config, 877)
        state, key = backend.state, backend.key
        _, subkey = jax.random.split(key)
        model = core.SCP(5, 1.1)
        params = scs_paper._parameters(model, mapping)
        target = Target(scs_paper.target_spec(config))
        density = model.transform_target(lambda x: target.log_prob(x, 'jax'), params)
        states, accepted = core.rwm_bright_side_stepout(density, state, 1.1, subkey, 300, stepsize=.1)
        norms = np.linalg.norm(np.asarray(model.projection(params, states)), axis=1)
        prefix, saved = backend.batch()
        np.testing.assert_allclose(prefix, prefixes_from(norms, np.asarray(accepted)), rtol=1e-12)
        np.testing.assert_allclose(saved, norms[9::10], rtol=1e-12)
        np.testing.assert_allclose(backend.state, states[-1], atol=1e-12)
        self.assertEqual(backend.scale, .1)

    def test_invalid_protocol_inputs_fail_before_backend_creation(self):
        base = dict(algorithm='SCS', time_budget=.05, warmup=100, thin=10)
        for change in ({'time_budget': 0}, {'time_budget': np.inf}, {'thin': 3},
                       {'warmup': -1}, {'warmup': True}, {'algorithm': 'NUTS'}):
            with self.subTest(change=change), patch.object(timed, '_backend') as backend:
                with self.assertRaises(ValueError):
                    timed.run_timed({}, {**base, **change}, 1)
                backend.assert_not_called()


if __name__ == '__main__':
    unittest.main()
