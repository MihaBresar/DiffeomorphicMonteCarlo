"""Exact first-third exclusion, frozen kernels and equivalent batch statistics."""
import unittest
from unittest.mock import patch

import numpy as np

from dcs_bench import timed_radial as timed
from dcs_bench import scs_paper
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
        self.assertEqual(result['step_policy'], timed.STEP_POLICY)
        self.assertEqual(result['target_acceptance'], .234)
        self.assertEqual(result['common_cpu_backend'], 'JAX')
        self.assertEqual(result['acceptance'], .5)
        self.assertEqual(result['postburn_acceptance'], .5)

    def test_original_scs_full_scan_and_shared_recording_are_unchanged(self):
        scs_paper._author()
        import jax
        import jax.numpy as jnp
        config = dict(algorithm='SCS', dimension=5, nu=3., alpha={0: 3., 1: -3.},
                      thin=10, warmup=37)
        mapping = SphereMap(5, mu=np.linspace(-.3, .2, 5), scale=1.7,
                            observer=np.array([.1, -.05, 0., 0., 0.])).parameters()
        backend = timed._backend(mapping, config, 774)
        core = scs_paper._author()
        model = core.SCP(5, scs_paper.LATITUDE)
        params = scs_paper._parameters(model, mapping)
        target = Target(scs_paper.target_spec(config))
        density = model.transform_target(lambda x: target.log_prob(x, 'jax'), params)
        _, subkey = jax.random.split(backend.key)
        states, accepted = core.rwm_bright_side_stepout(density, backend.state,
            scs_paper.LATITUDE, subkey, nsample=300, stepsize=backend.scale)
        # Compare the whole shared scan, not just a derived aggregate.
        block, project, initial, _ = timed._components(mapping, target.spec, 'SCS')
        matched_states, matched_accepts = block(backend.state, subkey, 300, backend.scale)
        np.testing.assert_array_equal(matched_states, states)
        np.testing.assert_array_equal(matched_accepts, accepted)
        np.testing.assert_allclose(project(initial), np.ones(5), atol=1e-12)
        norms = np.linalg.norm(np.asarray(model.projection(params, states)), axis=1)
        frozen = backend.scale
        prefixes, saved = backend.batch()
        np.testing.assert_allclose(prefixes, prefixes_from(norms, accepted), rtol=1e-12)
        np.testing.assert_allclose(saved, norms[9::10], rtol=1e-12)
        np.testing.assert_allclose(backend.state, states[-1], rtol=1e-12)
        self.assertEqual(backend.scale, frozen)
        self.assertNotEqual(backend.scale, .1)
        self.assertTrue(0 <= backend.warmup_acceptance <= 1)

    def test_dcs_jax_proposal_density_and_acceptance_match_reference(self):
        scs_paper._author()
        import jax
        import jax.numpy as jnp
        target = Target(dict(kind='skew_t', dimension=5, nu=3., alpha={0: 3., 1: -3.}))
        mapping = BallMap(5, mu=np.linspace(-.3, .2, 5), scale=1.7,
                          delta=np.array([.1, -.05, 0., 0., 0.]))
        for algorithm in ('DCS-RWM', 'DCS-BW'):
            block, project, initial, density = timed._components(mapping.parameters(), target.spec, algorithm)
            np.testing.assert_allclose(project(initial), np.ones(5), atol=1e-12)
            for point in (np.zeros(5), np.asarray(initial), np.array([.99, 0., 0., 0., 0.])):
                self.assertAlmostEqual(float(density(jnp.asarray(point))), mapping.log_density(target, point), places=10)
            self.assertEqual(float(density(jnp.ones(5))), -np.inf)
            for scale in (.01, .3, 10.):
                with self.subTest(algorithm=algorithm, scale=scale):
                    key = jax.random.key(113)
                    key, subkey = jax.random.split(key)
                    direction = np.array(jax.random.normal(subkey, shape=initial.shape))
                    if algorithm == 'DCS-BW':
                        key, subkey = jax.random.split(key)
                        direction *= float(jax.random.uniform(subkey)) ** (1./5) / np.linalg.norm(direction)
                    proposed = np.asarray(initial) + scale * direction
                    logratio = mapping.log_density(target, proposed) - mapping.log_density(target, np.asarray(initial))
                    key, subkey = jax.random.split(key)
                    accepted = float(np.log(float(jax.random.uniform(subkey))) < logratio)
                    expected = proposed if accepted else np.asarray(initial)
                    states, accepts = block(initial, jax.random.key(113), 1, scale)
                    np.testing.assert_allclose(states[0], expected, rtol=1e-12, atol=1e-14)
                    self.assertEqual(float(accepts[0]), accepted)

    def test_shared_warmup_uses_observed_acceptance_and_smoothed_final_scale(self):
        scs_paper._author()
        import jax
        import jax.numpy as jnp
        outcomes = np.array([1., 0., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1.])
        # Save each scale in the synthetic state, so this checks every raw
        # warmup update as well as the final smoothed, frozen scale.
        def controlled_block(state, key, count, scale):
            index = state[0].astype(jnp.int32)
            state = state.at[index + 1].set(scale).at[0].add(1)
            return state[None, :], jnp.asarray(outcomes)[index][None]
        initial = jnp.zeros(len(outcomes) + 1)
        state, scale, accepts = timed._calibrate(controlled_block, initial, jax.random.key(2), len(outcomes), .1)
        raw, hbar, logbar, expected_scales = .1, 0., np.log(.1), []
        for iteration, accepted in enumerate(outcomes, 1):
            expected_scales.append(raw)
            eta = 1. / (iteration + 10.)
            hbar = (1 - eta) * hbar + eta * (.234 - accepted)
            lograw = np.log(10 * .1) - np.sqrt(iteration) / .05 * hbar
            weight = iteration ** (-.75)
            logbar = weight * lograw + (1 - weight) * logbar
            raw = np.exp(np.clip(lograw, -30, 15))
        np.testing.assert_allclose(state[1:], expected_scales, rtol=1e-12)
        np.testing.assert_allclose(float(scale), np.exp(logbar), rtol=1e-14)
        self.assertEqual(float(accepts), outcomes.sum())
        unchanged, untuned, accepts = timed._calibrate(controlled_block, initial, jax.random.key(2), 0, .1)
        np.testing.assert_array_equal(unchanged, initial)
        self.assertAlmostEqual(float(untuned), .1)
        self.assertEqual(float(accepts), 0.)

    def test_dcs_full_scan_shared_recording_and_frozen_step(self):
        scs_paper._author()
        import jax
        for algorithm in ('DCS-RWM', 'DCS-BW'):
            with self.subTest(algorithm=algorithm):
                config = dict(algorithm=algorithm, dimension=5, nu=3., alpha={0: 3.},
                              thin=10, warmup=23)
                mapping = BallMap(5, scale=1.7).parameters()
                backend = timed._backend(mapping, config, 774)
                block, projection, _, _ = timed._components(mapping, scs_paper.target_spec(config), algorithm)
                _, subkey = jax.random.split(backend.key)
                states, accepted = block(backend.state, subkey, 300, backend.scale)
                norms = np.linalg.norm(np.asarray(projection(states)), axis=1)
                frozen = backend.scale
                prefixes, saved = backend.batch()
                np.testing.assert_allclose(prefixes, prefixes_from(norms, accepted), rtol=1e-12)
                np.testing.assert_allclose(saved, norms[9::10], rtol=1e-12)
                np.testing.assert_allclose(backend.state, states[-1], rtol=1e-12)
                self.assertEqual(backend.scale, frozen)
                self.assertTrue(np.any(np.asarray(accepted) == 0))

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
