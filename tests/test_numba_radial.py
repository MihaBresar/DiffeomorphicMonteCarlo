"""Native DCS parity against independent target/map and state-by-state oracles."""
from types import SimpleNamespace
import unittest

import numpy as np

from dcs_bench import numba_radial as native
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap


def direct_log_prob(spec, point):
    return Target(spec).log_prob(point)


def reflected_reference(state, increment):
    """Scalar line-coordinate construction, independent of native scratch loops."""
    raw = state + increment
    if np.linalg.norm(raw) < 1.:
        return raw
    unit = increment / np.linalg.norm(increment)
    position = state @ unit
    transverse = state - position * unit
    half = np.sqrt(1. - state @ state + position * position)
    chord = 2. * half
    folded = chord - abs(((position + half + np.linalg.norm(increment)) % (2. * chord)) - chord) - half
    return transverse + folded * unit


class NumbaRadialTests(unittest.TestCase):
    def test_density_and_norm_match_direct_target_plus_jacobian(self):
        rng = np.random.default_rng(842)
        for d in (5, 100, 500):
            for kind, beta in (('skew_t', 2.), ('skew_t', 1.)):
                for fitted in (False, True):
                    with self.subTest(d=d, kind=kind, beta=beta, fitted=fitted):
                        spec = dict(kind=kind, dimension=d, nu=3., alpha={'0': 8., '1': -3.})
                        delta = rng.normal(size=d)
                        delta *= .25 / np.linalg.norm(delta) if fitted else 0.
                        mapping = BallMap(d, beta=beta, mu=np.linspace(-.1, .2, d) if fitted else None,
                                          scale=1.9 if fitted else 1., delta=delta)
                        args = native._inputs(mapping.parameters(), {'target': spec})
                        for radius in (0., .5, .99):
                            direction = rng.normal(size=d)
                            state = direction * radius / np.linalg.norm(direction)
                            point, jacobian = mapping.forward(state)
                            expected = direct_log_prob(spec, point) + jacobian
                            observed, norm = native._evaluate(state, *args)
                            self.assertAlmostEqual(observed, expected, delta=2e-8)
                            np.testing.assert_allclose(norm, np.linalg.norm(point), rtol=2e-12, atol=2e-13)

    def test_beta_two_boundary_identity_remains_finite_near_boundary(self):
        mapping = BallMap(100, scale=1.7, delta=np.r_[.3, np.zeros(99)])
        spec = dict(kind='skew_t', dimension=100, nu=3., alpha={'0': 100., '1': -100.})
        args = native._inputs(mapping.parameters(), {'target': spec})
        state = np.r_[np.nextafter(1., 0.), np.zeros(99)]
        value, radius = native._evaluate(state, *args)
        self.assertTrue(np.isfinite(value))
        self.assertTrue(np.isfinite(radius))
        for outside in (np.ones(100), np.full(100, np.nan)):
            self.assertEqual(native._evaluate(outside, *args)[0], -np.inf)

    def test_reflection_endpoints_match_jax_at_multiple_chord_crossings(self):
        from dcs_bench import scs_paper
        from dcs_bench.ball_boundary import reflect_ball_proposal
        scs_paper._author()
        import jax
        import jax.numpy as jnp
        rng = np.random.default_rng(7492)
        for d in (3, 100):
            oracle = jax.jit(reflect_ball_proposal)
            for radius in (.0, .8, .99999):
                state = rng.normal(size=d)
                state *= radius / np.linalg.norm(state)
                for length in (.01, .2, 2., 50., 10000.):
                    increment = rng.normal(size=d)
                    increment *= length / np.linalg.norm(increment)
                    expected, valid = oracle(jnp.asarray(state), jnp.asarray(increment))
                    result = np.empty(d)
                    observed = native._proposal_into(state, increment, True, result)
                    with self.subTest(d=d, radius=radius, length=length):
                        self.assertEqual(observed, bool(valid))
                        if observed:
                            np.testing.assert_allclose(result, expected, atol=3e-9, rtol=2e-9)
                            self.assertLess(result @ result, 1.)
        scratch = np.empty(2)
        self.assertFalse(native._proposal_into(np.array([.9, 0.]), np.array([.2, 0.]), False, scratch))

    def test_controlled_metropolis_trials_include_density_ratio_and_rejection_cache(self):
        rng = np.random.default_rng(3274)
        for kind, beta in (('skew_t', 2.), ('skew_t', 1.)):
            spec = dict(kind=kind, dimension=5, nu=3., alpha={'0': 9.})
            mapping = BallMap(5, beta=beta, scale=1.5, mu=np.linspace(-.1, .1, 5),
                              delta=np.array([.1, -.05, 0., 0., 0.]))
            args = native._inputs(mapping.parameters(), {'target': spec})
            for reflected in (False, True):
                state = mapping.inverse(np.ones(5))
                lp, radius = native._evaluate(state, *args)
                for trial in range(50):
                    increment = rng.normal(size=5) * (.07 if trial % 2 else 1.3)
                    logu = np.log(rng.random())
                    candidate = reflected_reference(state, increment) if reflected else state + increment
                    valid = candidate @ candidate < 1.
                    before = state.copy()
                    if valid:
                        x, jac = mapping.forward(candidate)
                        proposed_lp = direct_log_prob(spec, x) + jac
                    else:
                        proposed_lp = -np.inf
                    accepted = logu < proposed_lp - lp
                    old_lp, old_radius = lp, radius
                    lp, radius, observed = native._transition_from_increment(
                        state, lp, radius, increment, logu, reflected, *args, np.empty(5))
                    self.assertEqual(observed, accepted)
                    np.testing.assert_allclose(state, candidate if accepted else before, atol=2e-13)
                    if accepted:
                        self.assertAlmostEqual(lp, proposed_lp, places=10)
                        self.assertAlmostEqual(radius, np.linalg.norm(x), places=10)
                    else:
                        self.assertEqual(lp, old_lp)
                        self.assertEqual(radius, old_radius)

    def test_all_transition_statistics_match_independent_numpy_trajectory(self):
        for reflected in (False, True):
            for kind, beta in (('skew_t', 2.), ('skew_t', 1.)):
                with self.subTest(reflected=reflected, kind=kind):
                    spec = dict(kind=kind, dimension=5, nu=3., alpha={'0': 4.})
                    mapping = BallMap(5, beta=beta, scale=1.7)
                    args = native._inputs(mapping.parameters(), {'target': spec})
                    initial = mapping.inverse(np.ones(5))
                    state = initial.copy()
                    physical, jacobian = mapping.forward(state)
                    lp = direct_log_prob(spec, physical) + jacobian
                    norm = np.linalg.norm(physical)
                    rng = np.random.default_rng(1643)
                    radii, accepts = [], []
                    for i in range(300):
                        normals = rng.standard_normal(7)
                        increment = .5 * normals[:5] / np.linalg.norm(normals)
                        logu = np.log(rng.random())
                        proposed = reflected_reference(state, increment) if reflected else state + increment
                        if proposed @ proposed < 1.:
                            point, jac = mapping.forward(proposed)
                            new_lp = direct_log_prob(spec, point) + jac
                        else:
                            new_lp = -np.inf
                        accepted = logu < new_lp - lp
                        if accepted:
                            state, lp, norm = proposed, new_lp, np.linalg.norm(point)
                        radii.append(norm)
                        accepts.append(accepted)
                    native_state = initial.copy()
                    native_lp, native_norm = native._evaluate(native_state, *args)
                    observed_lp, observed_norm, prefixes, saved = native._run_block(
                        np.random.default_rng(1643), native_state, native_lp, native_norm,
                        .5, 300, 10, 5., reflected, *args)
                    radii = np.asarray(radii)
                    expected = np.cumsum(np.column_stack([radii >= 5., radii, accepts]), axis=0)[99::100]
                    # Repeated chord folds amplify tiny reduction-order
                    # differences over a full 300-transition batch. Decisions/counts must
                    # still agree exactly; one-step geometry is tested tightly.
                    np.testing.assert_array_equal(prefixes[:, (0, 2)], expected[:, (0, 2)])
                    np.testing.assert_allclose(prefixes[:, 1], expected[:, 1], rtol=2e-10, atol=1e-10)
                    np.testing.assert_allclose(saved, radii[9::10], rtol=2e-10, atol=1e-10)
                    np.testing.assert_allclose(native_state, state, rtol=2e-10, atol=1e-10)
                    np.testing.assert_allclose(observed_lp, lp, rtol=2e-10, atol=1e-10)
                    np.testing.assert_allclose(observed_norm, norm, rtol=2e-10, atol=1e-10)

    def test_backend_reuses_calibration_without_mutating_it_and_freezes_scale(self):
        mapping = BallMap(5, scale=1.7).parameters()
        state = BallMap(**mapping).inverse(np.ones(5))
        history = {'tuning_history': [{'scale': .4, 'selected': True}]}
        calibration = SimpleNamespace(state=state, scale=.4, compile_seconds=3., warmup_seconds=2.,
            warmup_acceptance=.8, initial_scale=.1, tuning_policy='ball_esjd_pilot_v1', tuning_metadata=history)
        config = dict(target=dict(kind='skew_t', dimension=5, nu=3., alpha={'0': 3.}),
                      algorithm='DCS-BW-reflect', thin=10, threshold=5.)
        first = native.NumbaRadial(mapping, config, 71, calibration)
        second = native.NumbaRadial(mapping, config, 71, calibration)
        np.testing.assert_array_equal(first.state, state)
        self.assertEqual(first.warmup_seconds, 2.)
        self.assertEqual(first.warmup_acceptance, .8)
        self.assertEqual(first.scale, .4)
        self.assertEqual(first.tuning_metadata['tuning_history'], history['tuning_history'])
        for _ in range(2):
            a, b = first.batch(), second.batch()
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
            self.assertEqual(a[0].shape, (3, 3))
            self.assertEqual(a[1].shape, (30,))
            self.assertEqual(first.scale, .4)
        np.testing.assert_array_equal(calibration.state, state)
        self.assertNotIn('native_compile_seconds', history)


if __name__ == '__main__':
    unittest.main()
