"""Independent target, radial-reference and transform checks for both suites."""
import unittest

import numpy as np
from scipy import stats

from dcs_bench.comparison_targets import TARGET_IDS, comparison_target
from dcs_bench.references import radial_truth
from dcs_bench.targets import Target

class ComparisonTargetTests(unittest.TestCase):
    def test_target_registry_is_explicit_and_returns_independent_data(self):
        self.assertEqual(TARGET_IDS, ('skew-t-100d', 'skew-t-500d'))
        old = comparison_target('skew-t-100d')
        large = comparison_target('skew-t-500d')
        self.assertEqual(old['target']['dimension'], 100)
        self.assertEqual(old['threshold'], 50.)
        self.assertEqual(old['coordinate_threshold'], 3.)
        self.assertEqual(large['target']['dimension'], 500)
        self.assertEqual(large['target']['nu'], 3.)
        self.assertEqual(large['target']['alpha'], {'0': 100., '1': -100.})
        self.assertEqual(large['beta'], 2.)
        self.assertAlmostEqual(large['threshold'], 50.*np.sqrt(5.))
        self.assertEqual(large['coordinate_threshold'], 3.)
        large['target']['alpha']['0'] = -1
        self.assertEqual(comparison_target('skew-t-500d')['target']['alpha']['0'], 100.)
        with self.assertRaises(ValueError):
            comparison_target('unknown')

    def test_student_500_reference_uses_500_dimensions(self):
        entry = comparison_target('skew-t-500d')
        target = Target(entry['target'])
        _, truth, _ = radial_truth(target, entry['threshold'])
        radial_second = target.nu*target.d/(target.nu-2.)
        self.assertGreater(truth[1], radial_truth(Target(comparison_target('skew-t-100d')['target']), 50.)[1][1])
        self.assertLess(truth[1]**2, radial_second)
        self.assertAlmostEqual(truth[0], stats.f.sf(entry['threshold']**2/500, 500, 3.))
        point = np.ones(500)
        self.assertTrue(np.isfinite(target.log_prob(point)))

if __name__ == '__main__':
    unittest.main()
