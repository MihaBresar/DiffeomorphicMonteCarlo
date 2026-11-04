"""Native mixture law, Mobius correction and all-state recording checks."""
from types import SimpleNamespace
import unittest

import numpy as np

from dcs_bench import numba_radial as radial
from dcs_bench import numba_rotation as native
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, mobius
from tests.test_numba_asymmetry import direct_log_prob
from tests.test_numba_radial import reflected_reference


def spec(d):
    return dict(kind='skew_t', dimension=d, nu=3., alpha={'0': 8., '1': -3.})


def mapping(d, fitted):
    return BallMap(d, scale=1.6 if fitted else 1.,
                   mu=np.linspace(-.15, .25, d) if fitted else None,
                   delta=np.r_[.2, -.1, np.zeros(d-2)] if fitted else None)


def rotation_oracle(z, noise, scale, delta):
    w, old_logmo = mobius(z, delta)
    radius = np.linalg.norm(w)
    u = w/radius
    tangent = noise-u*np.dot(noise, u)
    tangent /= np.linalg.norm(tangent)
    theta = 2*np.arctan(scale)
    next_w = np.cos(theta)*w+np.sin(theta)*radius*tangent
    proposed = mobius(next_w, -delta)[0]
    new_logmo = mobius(proposed, delta)[1]
    return proposed, old_logmo-new_logmo


class NumbaRotationTests(unittest.TestCase):
    def test_jax_proposal_and_physical_mh_ratio_match_with_and_without_vi(self):
        from dcs_bench import scs_paper
        scs_paper._author()
        from dcs_bench.rotation import angular_proposal
        import jax.numpy as jnp

        rng = np.random.default_rng(24091)
        for d in (6, 100, 500):
            for fitted in (False, True):
                transform = mapping(d, fitted)
                target = Target(spec(d))
                for radius in (.2, .9, .9999):
                    w = rng.normal(size=d)
                    w *= radius/np.linalg.norm(w)
                    z = mobius(w, -transform.delta)[0]
                    for scale in (.2, 1., 3.):
                        normal = rng.normal(size=d)
                        candidate, correction, valid = native.angular_proposal(z, normal, scale, transform.delta)
                        jax_candidate, jax_correction, jax_valid = angular_proposal(
                            jnp.asarray(z), jnp.asarray(normal), scale, jnp.asarray(transform.delta))
                        with self.subTest(d=d, fitted=fitted, radius=radius, scale=scale):
                            self.assertTrue(valid)
                            self.assertTrue(bool(jax_valid))
                            np.testing.assert_allclose(candidate, jax_candidate, rtol=2e-12, atol=2e-13)
                            self.assertAlmostEqual(correction, float(jax_correction), delta=2e-10)
                            x, jac = transform.forward(z)
                            next_x, next_jac = transform.forward(candidate)
                            self.assertAlmostEqual(np.linalg.norm(x-transform.mu),
                                                   np.linalg.norm(next_x-transform.mu),
                                                   delta=2e-8)
                            physical_ratio = direct_log_prob(spec(d), next_x)-direct_log_prob(spec(d), x)
                            corrected_ratio = transform.log_density(target, candidate)-transform.log_density(target, z)+correction
                            self.assertAlmostEqual(corrected_ratio, physical_ratio, delta=2e-7)
                            # An uncorrected raw-ball MH ratio is wrong for a
                            # nontrivial Mobius deformation; correction equals
                            # the inverse proposal-Jacobian contribution.
                            self.assertAlmostEqual(next_jac-jac+correction, 0., delta=2e-7)

    def test_reverse_rotation_recovers_state_and_opposite_correction(self):
        rng = np.random.default_rng(2981)
        transform = mapping(9, True)
        z = transform.inverse(np.linspace(-2., 2., 9))
        for scale in (.2, 1., 3.):
            proposed, correction, valid = native.angular_proposal(z, rng.normal(size=9), scale, transform.delta)
            self.assertTrue(valid)
            w, next_w = mobius(z, transform.delta)[0], mobius(proposed, transform.delta)[0]
            theta = 2*np.arctan(scale)
            reverse_noise = (w-np.cos(theta)*next_w)/(np.sin(theta)*np.linalg.norm(next_w))
            reverse, reverse_correction, reverse_valid = native.angular_proposal(
                proposed, reverse_noise, scale, transform.delta)
            self.assertTrue(reverse_valid)
            np.testing.assert_allclose(reverse, z, rtol=3e-12, atol=3e-13)
            self.assertAlmostEqual(reverse_correction, -correction, places=11)

    def test_invalid_angular_proposals_leave_a_self_loop_without_clipping(self):
        zero = np.zeros(6)
        state = np.r_[.5, np.zeros(5)]
        for point, normal, scale in ((zero, np.ones(6), 1.),
                                      (state, state.copy(), 1.),
                                      (state, np.full(6, np.nan), 1.),
                                      (state, np.ones(6), 0.),
                                      (state, np.ones(6), np.inf),
                                      (np.ones(6), np.ones(6), 1.)):
            proposal, correction, valid = native.angular_proposal(point, normal, scale, zero)
            self.assertFalse(valid)
            self.assertEqual(correction, 0.)
            np.testing.assert_array_equal(proposal, point)

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
                    normal = rng.standard_normal(d+2)
                    logu = np.log(rng.random())
                    if angular:
                        proposed, correction = rotation_oracle(state, normal[:d], .8, transform.delta)
                    else:
                        proposed = reflected_reference(state, .4*normal[:d]/np.linalg.norm(normal))
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
                logmo = native._mobius_into(initial, inputs[1], inputs[3][3], inputs[3][5], w)
                for thin in (1, 10, 50):
                    native_state, native_w = initial.copy(), w.copy()
                    output = native._run_block(np.random.default_rng(48933), native_state, native_w,
                        *initial_values, logmo, .4, .8, 300, thin, 5., 1.25, *inputs)
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
                logmo = native._mobius_into(initial, inputs[1], inputs[3][3], inputs[3][5], w)
                state = initial.copy()
                rng = np.random.default_rng(381)
                full = native._run_block(rng, state, w.copy(), *initial_values, logmo,
                    .4, .8, 900, 1, 5., 1.25, *inputs)
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
                            .4, .8, 900, thin, 5., 1.25, *inputs)
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
        base = dict(algorithm='DCS-BW-rotate', target=spec(6), thin=10,
                    threshold=5., coordinate_threshold=1.25)
        extended = native.NumbaRotation(transform.parameters(), dict(base, coordinate_metrics=True), 729, calibration)
        radial_only = native.NumbaRotation(transform.parameters(), dict(base, coordinate_metrics=False), 729, calibration)
        for _ in range(2):
            a, b = extended.batch(), radial_only.batch()
            self.assertEqual(a[0].shape, (3, 9))
            self.assertEqual(a[1].shape, (30, 4))
            np.testing.assert_array_equal(a[0][:, :3], b[0])
            np.testing.assert_array_equal(a[1][:, 0], b[1])
            np.testing.assert_array_equal(extended.state, radial_only.state)
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
        radial2 = native.NumbaRotation(transform2.parameters(), config2, 729, calibration2)
        prefixes2, saved2 = radial2.batch()
        self.assertEqual(prefixes2.shape, (3, 3))
        self.assertEqual(saved2.shape, (30,))
        self.assertTrue(np.all(np.isfinite(prefixes2)))
        self.assertTrue(np.all(np.isfinite(saved2)))
        with self.assertRaises(ValueError):
            native.NumbaRotation(transform2.parameters(), dict(config2, coordinate_metrics=True), 729, calibration2)


if __name__ == '__main__':
    unittest.main()
