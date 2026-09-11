"""Scientific parity checks for canonical native kernels and their NumPy oracle.

The reference target, maps and MetropolisKernel are independently implemented;
these checks do not construct expected values from native implementation helpers.
"""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from dcs_bench import compiled_sampling as native
from dcs_bench.samplers import MetropolisKernel, run_numpy_metropolis
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap


def targets():
    return [
        Target(dict(kind='skew_t', dimension=5, nu=3., alpha={'0': 7., '1': -3.})),
        Target(dict(kind='skew_t', dimension=50, nu=.5, alpha={})),
        Target(dict(kind='logistic', dimension=5, nu=1., prior_scale=2.5,
                    n_observations=30, data_seed=42)),
        Target(dict(kind='eight_schools', dimension=10, parameterization='centered')),
        Target(dict(kind='eight_schools', dimension=10, parameterization='noncentered')),
        Target(dict(kind='garch', dimension=4)),
    ]


def label(target):
    return (target.kind, target.spec.get('parameterization'), target.spec.get('nu'))


def ball(target):
    shift = np.linspace(-.08, .09, target.d) / np.sqrt(target.d)
    beta = .2 if target.kind == 'skew_t' and target.nu == .5 else 2.
    return BallMap(target.d, beta=beta, mu=.15*target.initial(), scale=2.3, delta=shift)


def sphere(target):
    return SphereMap(target.d, mu=.15*target.initial(), scale=1.7,
                     observer=np.linspace(-.07, .09, target.d)/np.sqrt(target.d), latitude=1.1)


def method_maps(target):
    bounded = ball(target)
    return [('DCS-RWM', bounded), ('DCS-BW', bounded), ('DCS-HnR', bounded),
            ('RWM-Gaussian', None), ('RWM-stable', None), ('SCS', sphere(target))]


class CompiledSamplingTests(unittest.TestCase):
    def test_all_canonical_densities_and_physical_coordinates(self):
        rng = np.random.default_rng(91541)
        for target in targets():
            for mapping in (None, ball(target), sphere(target)):
                _, context = native._prepare(target, mapping)
                kind, mapped, mu, delta, alpha, constants, parameters, data = context
                physical_points = target.initial() + rng.normal(0., .3, (10, target.d))
                for x in physical_points:
                    z = x if mapping is None else mapping.inverse(x)
                    with self.subTest(target=label(target), mapping=type(mapping).__name__):
                        expected_density = target.log_prob(x)
                        if mapping is not None:
                            decoded, logjac = mapping.forward(z)
                            # Include map roundoff in both reference formulas.
                            expected_density = target.log_prob(decoded) + logjac
                            assert_allclose(decoded, x, rtol=4e-12, atol=3e-11)
                        actual = native._density(z, *context)
                        self.assertTrue(np.isfinite(actual))
                        assert_allclose(actual, expected_density, rtol=3e-12, atol=3e-10)
                        decoded = native._physical(z, kind, mapped, mu, delta, constants, parameters)
                        assert_allclose(decoded, target.physical(x), rtol=5e-12, atol=3e-10)

    def test_superheavy_ball_density_remains_stable_at_boundary(self):
        target = Target(dict(kind='skew_t', dimension=50, nu=.5, alpha={}))
        mapping = BallMap(50, beta=.2)
        _, context = native._prepare(target, mapping)
        for radius in (0., .2, .8, .999, 1.-1e-8):
            z = np.zeros(50); z[0] = radius
            with self.subTest(radius=radius):
                actual = native._density(z, *context)
                self.assertTrue(np.isfinite(actual))
                assert_allclose(actual, mapping.log_density(target, z), rtol=3e-12, atol=3e-10)
        for radius in (1., 1.01, 2.):
            z = np.zeros(50); z[0] = radius
            self.assertEqual(native._density(z, *context), -np.inf)

    def test_one_step_matches_numpy_state_acceptance_work_and_random_state(self):
        outcomes = {name: set() for name in native.ALGORITHMS}
        slice_evaluations = []
        for target in targets():
            for algorithm, mapping in method_maps(target):
                initial, context = native._prepare(target, mapping)
                initial_lp = native._density(initial, *context)
                for index in range(12):
                    seed, scale = 17560+index, (.02, .15, 1.5)[index % 3]
                    rng = np.random.default_rng(seed)
                    oracle = MetropolisKernel(target, algorithm, mapping, seed)
                    evaluations_before = oracle.evaluations
                    expected_probability, expected_accept = oracle.step(scale)
                    actual = native._transition(rng, initial.copy(), initial_lp, scale,
                                                native.ALGORITHMS[algorithm], *context)
                    state, density, probability, accepted, evaluations = actual
                    with self.subTest(target=label(target), algorithm=algorithm, seed=seed):
                        self.assertEqual(bool(accepted), expected_accept)
                        self.assertEqual(evaluations, oracle.evaluations-evaluations_before)
                        assert_allclose(state, oracle.state, rtol=3e-11, atol=3e-11)
                        assert_allclose(density, oracle.lp, rtol=3e-11, atol=3e-9)
                        assert_allclose(probability, expected_probability, rtol=3e-10, atol=3e-11)
                        self.assertEqual(rng.bit_generator.state, oracle.rng.bit_generator.state)
                    outcomes[algorithm].add(bool(accepted))
                    if algorithm == 'DCS-HnR':
                        slice_evaluations.append(evaluations)
        for algorithm, accepted in outcomes.items():
            if algorithm != 'DCS-HnR':
                self.assertEqual(accepted, {False, True}, algorithm)
        self.assertGreater(max(slice_evaluations), 1)

    def test_raw_warmup_and_frozen_short_runs_match_numpy(self):
        config = dict(warmup=20, iterations=60, thin=3, initial_scale=.1)
        for target in targets():
            for algorithm, mapping in method_maps(target):
                with self.subTest(target=label(target), algorithm=algorithm):
                    actual, info = native.run_compiled(target, algorithm, mapping, config, seed=19022)
                    expected, reference = run_numpy_metropolis(target, algorithm, mapping, config, seed=19022)
                    assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)
                    assert_allclose(info['frozen_scale'], reference['frozen_scale'], rtol=1e-9, atol=1e-12)
                    for key in ('iterations', 'retained_draws', 'warmup_iterations',
                                'log_density_evaluations', 'warmup_log_density_evaluations',
                                'acceptance_rate', 'adaptation_during_sampling'):
                        self.assertEqual(info[key], reference[key], key)
                    self.assertFalse(info['adaptation_during_sampling'])


if __name__ == '__main__':
    unittest.main()
