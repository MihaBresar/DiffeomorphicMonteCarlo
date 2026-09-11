"""Scientific labels stay readable without erasing small values or counts."""
import copy
import unittest

import numpy as np

from dcs_bench.display import number, cell


class DisplayTests(unittest.TestCase):
    def test_errors_have_two_scientific_decimals_and_tiny_values_survive(self):
        self.assertEqual(number(2.0956641753087117e-7, scientific=True), '2.10e-07')
        self.assertEqual(number(.148123, scientific=True), '1.48e-01')
        self.assertEqual(number(.000075123), '7.51e-05')
        self.assertEqual(number(33.201234), '33.20')
        self.assertEqual(number(.23456789), '0.23')
        self.assertEqual(number(-.00002345), '-2.35e-05')

    def test_actual_integer_counts_remain_exact(self):
        self.assertEqual(number(500000), '500000')
        self.assertEqual(number(np.int64(25000000)), '25000000')
        self.assertEqual(number(500000., integer=True), '500000')
        self.assertEqual(cell(25000000, 'total_iterations'), '25000000')
        self.assertEqual(cell(500000., 'mean_iterations'), '500000')
        self.assertEqual(cell(500000.125, 'mean_iterations'), '500000.12')

    def test_html_formatting_does_not_mutate_numerical_values(self):
        data = {'mse': [2.0956641753087117e-7, .148123], 'steps': [.23456789, .000075123], 'iterations': 500000}
        before = copy.deepcopy(data)
        rendered = cell(data)
        self.assertIn('2.10e-07', rendered)
        self.assertIn('1.48e-01', rendered)
        self.assertIn('7.51e-05', rendered)
        self.assertIn('500000', rendered)
        self.assertEqual(data, before)


if __name__ == '__main__':
    unittest.main()
