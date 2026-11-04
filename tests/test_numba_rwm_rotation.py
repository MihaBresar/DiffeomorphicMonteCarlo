"""Reflected Gaussian mixture law, shared calibration and all-state summaries."""
from types import SimpleNamespace
import unittest

import numpy as np

from dcs_bench import numba_radial as radial
from dcs_bench import numba_rotation as shared
from dcs_bench import numba_rwm_rotation as native
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, mobius
from dcs_bench.references import skew_t_iid
from tests.test_numba_asymmetry import direct_log_prob
from tests.test_numba_radial import reflected_reference
from tests.test_numba_rotation import spec, mapping, rotation_oracle


class NumbaRWMRotationTests(unittest.TestCase):
    def test_jax_single_transition_matches_native_helpers_and_physical_mh(self):
        from dcs_bench import scs_paper
        scs_paper._author()
        from dcs_bench import rwm_rotation
        import jax
        import jax.numpy as jnp
        for d in (6, 100, 500):
            for fitted in (False, True):
                transform = mapping(d, fitted)
                target = spec(d)
                block, _, _, _ = rwm_rotation.build_components(transform.parameters(), target)
                initial = transform.inverse(np.linspace(-1., 2., d))
                inputs = radial._inputs(transform.parameters(), {'target': target})
                old_lp = radial._evaluate_extended(initial, *inputs)[0]
                sigma, angle = .4/np.sqrt(d+2), .8
                program = jax.jit(lambda z, key: block(z, key, 1, jnp.array([sigma, angle])))
                selections = set()
                for seed in range(4):
                    key = jax.random.key(401+seed)
                    _, select_key, normal_key, accept_key = jax.random.split(key, 4)
                    angular = bool(jax.random.bernoulli(select_key, .5))
                    selections.add(angular)
                    normal = np.asarray(jax.random.normal(normal_key, (d,)))
                    logu = float(jnp.log(jax.random.uniform(accept_key)))
                    if angular:
                        candidate, correction, valid = shared.angular_proposal(initial, normal, angle, transform.delta)
                    else:
                        candidate = np.empty(d)
                        valid = radial._proposal_into(initial, sigma*normal, True, candidate)
                        correction = 0.
                    self.assertTrue(valid)
                    next_lp = radial._evaluate_extended(candidate, *inputs)[0]
                    expected_accept = logu < next_lp-old_lp+correction
                    physical, jac = transform.forward(initial)
                    proposed_physical, proposed_jac = transform.forward(candidate)
                    oracle_ratio = (direct_log_prob(target, proposed_physical)+proposed_jac
                                    - direct_log_prob(target, physical)-jac+correction)
                    self.assertAlmostEqual(next_lp-old_lp+correction, oracle_ratio, delta=2e-8)
                    states, accepted = program(jnp.asarray(initial), key)
                    with self.subTest(d=d, fitted=fitted, seed=seed):
                        self.assertEqual(bool(accepted[0]), expected_accept)
                        np.testing.assert_allclose(states[0], candidate if expected_accept else initial,
                                                   rtol=3e-11, atol=3e-12)
                self.assertEqual(selections, {True, False})

    def test_native_mixture_preserves_stationary_target_with_and_without_vi(self):
        # Independent stationary starts avoid interpreting a single finite
        # trajectory's apparent stability as evidence of invariance.
        d, count = 6, 3000
        target = Target(spec(d))
        physical = skew_t_iid(target, np.random.default_rng(70193), count)
        def features(x):
            radius = np.linalg.norm(x, axis=1)
            return np.column_stack((np.tanh(x[:, :3]), np.log1p(radius), x[:, 0] > 1.25))
        before = features(physical)
        for fitted in (False, True):
            transform = mapping(d, fitted)
            inputs = radial._inputs(transform.parameters(), {'target': spec(d)})
            states = transform.inverse(physical)
            output = np.empty_like(physical)
            rng = np.random.default_rng(91753)
            for i, initial in enumerate(states):
                state = initial.copy()
                values = radial._evaluate_extended(state, *inputs)
                w = np.empty(d)
                logmo = shared._mobius_into(state, inputs[1], inputs[3][3], inputs[3][5], w)
                native._run_block(rng, state, w, *values, logmo,
                                  .16, .8, 1, 1, 5., 1.25, *inputs)
                output[i] = transform.forward(state)[0]
            delta = features(output)-before
            error = delta.std(axis=0, ddof=1)/np.sqrt(count)
            np.testing.assert_array_less(np.abs(delta.mean(axis=0)), 6*error+1e-12)

    def test_seeded_full_mixture_matches_independent_physical_oracle(self):
        # Chord folds can amplify tiny reduction-order differences over a long
        # high-dimensional trajectory. One-step laws are checked at d=100/500
        # above; this independently evaluated complete path uses d=6.
        for d in (6,):
            for fitted in (False, True):
                transform = mapping(d, fitted)
                target = spec(d)
                initial = transform.inverse(np.ones(d))
                state = initial.copy()
                physical, jac = transform.forward(state)
                lp = direct_log_prob(target, physical)+jac
                rng = np.random.default_rng(48933)
                values, selections = [], []
                for _ in range(300):
                    angular = rng.random() < .5
                    normal = rng.standard_normal(d)
                    logu = np.log(rng.random())
                    if angular:
                        proposed, correction = rotation_oracle(state, normal[:d], .8, transform.delta)
                    else:
                        proposed = reflected_reference(state, .16*normal)
                        correction = 0.
                    new_physical, new_jac = transform.forward(proposed)
                    new_lp = direct_log_prob(target, new_physical)+new_jac
                    accepted = logu < new_lp-lp+correction
                    if accepted:
                        state, physical, lp = proposed, new_physical, new_lp
                    radius = np.linalg.norm(physical)
                    values.append([radius >= 5., radius, accepted, physical[0], physical[0] > 1.25,
                                   physical[1], physical[1] < -1.25, physical[2], physical[2] > 1.25])
                    selections.append(angular)
                self.assertGreater(sum(selections), 100)
                self.assertLess(sum(selections), 200)
                expected = np.asarray(values)
                inputs = radial._inputs(transform.parameters(), {'target': target})
                initial_values = radial._evaluate_extended(initial, *inputs)
                w = np.empty(d)
                logmo = shared._mobius_into(initial, inputs[1], inputs[3][3], inputs[3][5], w)
                for thin in (1, 10, 50):
                    native_state, native_w = initial.copy(), w.copy()
                    output = native._run_block(np.random.default_rng(48933), native_state, native_w,
                        *initial_values, logmo, .16, .8, 300, thin, 5., 1.25, *inputs)
                    prefixes, saved = output[-2:]
                    with self.subTest(d=d, fitted=fitted, thin=thin):
                        np.testing.assert_allclose(prefixes, expected.cumsum(axis=0)[99::100],
                                                   rtol=2e-10, atol=2e-7)
                        np.testing.assert_array_equal(prefixes[:, [0, 2, 4, 6, 8]],
                                                      expected.cumsum(axis=0)[99::100][:, [0, 2, 4, 6, 8]])
                        np.testing.assert_allclose(saved, expected[thin-1::thin][:, [1, 3, 5, 7]],
                                                   rtol=2e-9, atol=1e-7)
                        np.testing.assert_allclose(native_state, state, rtol=2e-10, atol=2e-10)
                        np.testing.assert_allclose(native_w, mobius(native_state, transform.delta)[0],
                                                   rtol=2e-12, atol=2e-13)
                        np.testing.assert_allclose(prefixes[-1]-prefixes[0], expected[100:].sum(axis=0),
                                                   rtol=2e-10, atol=2e-7)

    def test_high_dimensional_prefixes_and_thinning_match_all_native_states(self):
        for d in (100, 500):
            for fitted in (False, True):
                transform = mapping(d, fitted)
                initial = transform.inverse(np.ones(d))
                inputs = radial._inputs(transform.parameters(), {'target': spec(d)})
                initial_values = radial._evaluate_extended(initial, *inputs)
                w = np.empty(d)
                logmo = shared._mobius_into(initial, inputs[1], inputs[3][3], inputs[3][5], w)
                state = initial.copy()
                rng = np.random.default_rng(381)
                full = native._run_block(rng, state, w.copy(), *initial_values, logmo,
                    .4/np.sqrt(d+2), .8, 900, 1, 5., 1.25, *inputs)
                full_next_rng = rng.random(4)
                prefixes, saved = full[-2:]
                previous = np.vstack(([initial_values[1], *initial_values[2:]], saved[:-1]))
                accepted = np.any(saved != previous, axis=1)
                r, x1, x2, x3 = saved.T
                values = np.column_stack((r >= 5., r, accepted, x1, x1 > 1.25,
                                          x2, x2 < -1.25, x3, x3 > 1.25))
                with self.subTest(d=d, fitted=fitted):
                    np.testing.assert_array_equal(prefixes, values.cumsum(axis=0)[99::100])
                    np.testing.assert_allclose(prefixes[-1]-prefixes[2], values[300:].sum(axis=0),
                                               rtol=1e-12, atol=1e-9)
                    for thin in (10, 50):
                        thinned_state = initial.copy()
                        thinned_rng = np.random.default_rng(381)
                        thinned = native._run_block(thinned_rng, thinned_state, w.copy(), *initial_values, logmo,
                            .4/np.sqrt(d+2), .8, 900, thin, 5., 1.25, *inputs)
                        np.testing.assert_array_equal(thinned[-2], prefixes)
                        np.testing.assert_array_equal(thinned[-1], saved[thin-1::thin])
                        np.testing.assert_array_equal(thinned_state, state)
                        np.testing.assert_array_equal(thinned_rng.random(4), full_next_rng)

    def test_public_backend_reuses_calibration_and_records_both_metric_modes(self):
        transform = mapping(6, True)
        state = transform.inverse(np.ones(6))
        calibration = SimpleNamespace(state=state, scale=.4, angular_scale=.8,
            compile_seconds=1., warmup_seconds=2., warmup_acceptance=.7,
            tuning_metadata={'warmup_check': 60})
        base = dict(algorithm='DCS-RWM-rotate', target=spec(6), thin=10,
                    threshold=5., coordinate_threshold=1.25)
        extended = native.NumbaRWMRotation(transform.parameters(), dict(base, coordinate_metrics=True), 729, calibration)
        radial_only = native.NumbaRWMRotation(transform.parameters(), dict(base, coordinate_metrics=False), 729, calibration)
        for _ in range(2):
            a, b = extended.batch(), radial_only.batch()
            self.assertEqual(a[0].shape, (3, 9))
            self.assertEqual(a[1].shape, (30, 4))
            np.testing.assert_array_equal(a[0][:, :3], b[0])
            np.testing.assert_array_equal(a[1][:, 0], b[1])
            np.testing.assert_array_equal(extended.state, radial_only.state)
        self.assertEqual(extended.tuning_metadata['gaussian_step_sd'], calibration.scale)
        self.assertEqual(extended.tuning_metadata['gaussian_step_rms'], calibration.scale*np.sqrt(6))
        self.assertNotEqual(native.PRODUCTION_DOMAIN, shared.PRODUCTION_DOMAIN)
        self.assertEqual(extended.scale, calibration.scale)
        self.assertEqual(extended.angular_scale, calibration.angular_scale)
        self.assertEqual(extended.tuning_metadata['updates_per_iteration'], 1)
        self.assertEqual(calibration.tuning_metadata, {'warmup_check': 60})
        np.testing.assert_array_equal(calibration.state, state)

        # Radial-only comparisons need no third coordinate and support d=2.
        transform2 = mapping(2, True)
        calibration2 = SimpleNamespace(**vars(calibration))
        calibration2.state = transform2.inverse(np.ones(2))
        config2 = dict(base, target=spec(2), coordinate_metrics=False)
        radial2 = native.NumbaRWMRotation(transform2.parameters(), config2, 729, calibration2)
        prefixes2, saved2 = radial2.batch()
        self.assertEqual(prefixes2.shape, (3, 3))
        self.assertEqual(saved2.shape, (30,))
        self.assertTrue(np.all(np.isfinite(prefixes2)))
        self.assertTrue(np.all(np.isfinite(saved2)))
        with self.assertRaises(ValueError):
            native.NumbaRWMRotation(transform2.parameters(), dict(config2, coordinate_metrics=True), 729, calibration2)


if __name__ == '__main__':
    unittest.main()
