import unittest
import numpy as np
from dcs_bench.simulation_error import mse_variability, paired_mse_differences


class SimulationErrorTests(unittest.TestCase):
    def test_empirical_mcse_and_influential_chain_share(self):
        result = mse_variability([[1,1],[2,2],[3,3]], [0,0])
        np.testing.assert_allclose(result['mse_empirical_mcse'], [7/3,7/3])
        np.testing.assert_allclose(result['largest_chain_squared_error_share'], [9/14,9/14])
        self.assertEqual(mse_variability([[0,0]], [0,0]),
            dict(mse_empirical_mcse=[None,None],largest_chain_squared_error_share=[0.,0.]))

    def test_paired_differences_align_by_seed_not_row_order(self):
        means = {'DCS-BW': [[6,6],[2,2],[4,4]], 'SCS': [[1,1],[2,2],[3,3]]}
        records = {'DCS-BW': [{'seed':3},{'seed':1},{'seed':2}],
                   'SCS': [{'seed':1},{'seed':2},{'seed':3}]}
        rows = paired_mse_differences(means,records,np.array([0,0]))
        self.assertEqual(len(rows),2)
        for row in rows:
            self.assertAlmostEqual(row['mse_difference'],14.)
            self.assertAlmostEqual(row['difference_empirical_mcse'],7.)
        records['DCS-BW'][0]['seed'] = 2
        self.assertEqual(paired_mse_differences(means,records,np.array([0,0])),[])

    def test_coordinate_mse_differences_are_included(self):
        names = ['tail_probability', 'mean_norm', 'mean_x1', 'x1_tail_probability']
        rows = paired_mse_differences(
            {'DCS-BW': [[1, 2, 3, 4]], 'SCS': [[0, 0, 0, 0]]},
            {'DCS-BW': [{'seed': 7}], 'SCS': [{'seed': 7}]}, [0, 0, 0, 0], functionals=names)
        self.assertEqual([row['metric'] for row in rows], names)
        self.assertEqual([row['mse_difference'] for row in rows], [1., 4., 9., 16.])


if __name__ == '__main__':
    unittest.main()
