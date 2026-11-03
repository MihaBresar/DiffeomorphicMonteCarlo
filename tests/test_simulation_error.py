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


if __name__ == '__main__':
    unittest.main()
