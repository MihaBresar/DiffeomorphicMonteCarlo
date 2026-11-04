"""Independent coordinate references supplement the unchanged radial truths."""
import unittest

import numpy as np
from scipy import special, stats

from dcs_bench.comparison_targets import comparison_target
from dcs_bench.references import comparison_reference, radial_truth, skew_t_iid
from dcs_bench.targets import Target


class CoordinateReferenceTests(unittest.TestCase):
    def config(self, target_id='skew-t-100d', samples=12000):
        entry = comparison_target(target_id)
        return dict(target=entry['target'], threshold=entry['threshold'],
                    coordinate_threshold=entry['coordinate_threshold'], reference_samples=samples,
                    reference_seed=711, reference_batch_size=2048)

    def test_exact_radial_values_and_coordinate_reference_labels(self):
        for target_id in ('skew-t-100d', 'skew-t-500d'):
            config = self.config(target_id)
            result = comparison_reference(config)
            self.assertEqual(result['functionals'], ['tail_probability', 'mean_norm', 'mean_x1', 'x1_tail_probability'])
            expected = radial_truth(Target(config['target']), config['threshold'])[1]
            np.testing.assert_array_equal(result['truth'][:2], expected)
            self.assertEqual(result['reference_mcse'][:2], [0., 0.])
            self.assertEqual(result['exact'], [True, True, False, False])
            self.assertEqual(result['seed'], 711)
            self.assertTrue(np.all(np.isfinite(result['truth'])))
            self.assertGreater(result['truth'][2], .7)
            self.assertLess(result['truth'][2], .8)
            self.assertGreater(result['truth'][3], 0.)
            self.assertLess(result['truth'][3], .1)
            self.assertTrue(result['control_variates']['x1_tail_control_applied'])
            # A caller must not mutate the cached truth used by other methods.
            result['truth'][0] = -1.
            self.assertGreater(comparison_reference(config)['truth'][0], 0.)

    def test_coordinate_expectations_against_independent_exact_iid_draws(self):
        config = self.config(samples=40000)
        result = comparison_reference(config)
        draws = skew_t_iid(Target(config['target']), np.random.default_rng(723), 80000)
        values = np.column_stack((draws[:, 0], draws[:, 0] > config['coordinate_threshold']))
        raw_mean = values.mean(axis=0)
        joint_se = np.sqrt(values.var(axis=0, ddof=1)/len(values)+np.square(result['reference_mcse'][2:]))
        np.testing.assert_array_less(np.abs(raw_mean-result['truth'][2:]), 6.*joint_se)

    def test_symmetric_coordinate_mean_has_exact_anchor(self):
        config = self.config()
        config['target'] = dict(config['target'], alpha={})
        result = comparison_reference(config)
        self.assertTrue(result['exact'][2])
        self.assertEqual(result['truth'][2], 0.)
        self.assertEqual(result['reference_mcse'][2], 0.)
        self.assertFalse(result['control_variates']['x1_tail_control_applied'])
        expected = stats.t.sf(config['coordinate_threshold'], 3.)
        self.assertAlmostEqual(result['truth'][3], expected, delta=6.*result['reference_mcse'][3])

    def test_coordinate_mean_limit_is_absolute_t_moment(self):
        config = self.config()
        config['target'] = dict(config['target'], alpha={'0': 1e10})
        result = comparison_reference(config)
        expected = np.sqrt(3.)*np.exp(special.gammaln(1.)-special.gammaln(1.5))/np.sqrt(np.pi)
        self.assertAlmostEqual(result['truth'][2], expected, places=12)
        self.assertAlmostEqual(result['truth'][3], 2.*stats.t.sf(3., 3.), places=12)


if __name__ == '__main__':
    unittest.main()
