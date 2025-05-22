import unittest
import pandas as pd
from scaling_module import adjust_scale, normalize_scale

class TestScaling(unittest.TestCase):
    def setUp(self):
        # Set up a sample DataFrame for testing
        self.tbl = pd.DataFrame({
            'Height': [150, 160, 170, 180],
            'Weight': [50, 60, 70, 80]
        })

    def test_adjust_scale(self):
        # Test Adjustment (mean should be 0 and std should be 1)
        scaled_tbl = adjust_scale(self.tbl.copy(), ['Height', 'Weight'])
        self.assertAlmostEqual(scaled_tbl['Height'].mean(), 0.0, places=5, msg="Height column mean is not close to 0 after adjustment.")
        self.assertAlmostEqual(scaled_tbl['Weight'].mean(), 0.0, places=5, msg="Weight column mean is not close to 0 after adjustment.")
        self.assertAlmostEqual(scaled_tbl['Height'].std(), 1.0, places=5, msg="Height column standard deviation is not 1 after adjustment.")
        self.assertAlmostEqual(scaled_tbl['Weight'].std(), 1.0, places=5, msg="Weight column standard deviation is not 1 after adjustment.")

    def test_normalize_scale(self):
        # Test Normalization (values should be in the range [0, 1])
        scaled_tbl = normalize_scale(self.tbl.copy(), ['Height', 'Weight'])
        self.assertEqual(scaled_tbl['Height'].min(), 0.0, "Min value of Height after normalization should be 0.")
        self.assertEqual(scaled_tbl['Height'].max(), 1.0, "Max value of Height after normalization should be 1.")
        self.assertEqual(scaled_tbl['Weight'].min(), 0.0, "Min value of Weight after normalization should be 0.")
        self.assertEqual(scaled_tbl['Weight'].max(), 1.0, "Max value of Weight after normalization should be 1.")

    def test_adjust_empty_tbl(self):
        # Test Adjustment with an empty DataFrame
        empty_tbl = pd.DataFrame(columns=['Height', 'Weight'])
        scaled_tbl = adjust_scale(empty_tbl, ['Height', 'Weight'])
        self.assertTrue(scaled_tbl.empty, "The scaled DataFrame should be empty for an empty input.")

    def test_normalize_empty_tbl(self):
        # Test Normalization with an empty DataFrame
        empty_tbl = pd.DataFrame(columns=['Height', 'Weight'])
        scaled_tbl = normalize_scale(empty_tbl, ['Height', 'Weight'])
        self.assertTrue(scaled_tbl.empty, "The scaled DataFrame should be empty for an empty input.")

    def test_adjust_single_value_column(self):
        # Test Adjustment with a column that has only one unique value
        single_value_tbl = pd.DataFrame({'Height': [170, 170, 170, 170], 'Weight': [70, 70, 70, 70]})
        scaled_tbl = adjust_scale(single_value_tbl, ['Height', 'Weight'])
        self.assertEqual(scaled_tbl['Height'].std(), 0.0, "The standard deviation of a single-value column should be 0.")
        self.assertEqual(scaled_tbl['Weight'].std(), 0.0, "The standard deviation of a single-value column should be 0.")
        self.assertEqual(scaled_tbl['Height'].mean(), 0.0, "The mean should be 0 for a column with a single unique value after adjustment.")
        self.assertEqual(scaled_tbl['Weight'].mean(), 0.0, "The mean should be 0 for a column with a single unique value after adjustment.")

    def test_normalize_single_value_column(self):
        # Test Normalization with a column that has only one unique value
        single_value_tbl = pd.DataFrame({'Height': [170, 170, 170, 170], 'Weight': [70, 70, 70, 70]})
        scaled_tbl = normalize_scale(single_value_tbl, ['Height', 'Weight'])
        self.assertEqual(scaled_tbl['Height'].min(), 0.0, "Min value should be 0 when there is only one unique value.")
        self.assertEqual(scaled_tbl['Height'].max(), 0.0, "Max value should be 0 when there is only one unique value.")
        self.assertEqual(scaled_tbl['Weight'].min(), 0.0, "Min value should be 0 when there is only one unique value.")
        self.assertEqual(scaled_tbl['Weight'].max(), 0.0, "Max value should be 0 when there is only one unique value.")

    def test_adjust_with_negative_values(self):
        # Test Adjustment with negative values
        negative_tbl = pd.DataFrame({'Height': [-150, -160, -170, -180], 'Weight': [-50, -60, -70, -80]})
        scaled_tbl = adjust_scale(negative_tbl, ['Height', 'Weight'])
        self.assertAlmostEqual(scaled_tbl['Height'].mean(), 0.0, places=5, msg="Mean of negative values after adjustment should be 0.")
        self.assertAlmostEqual(scaled_tbl['Weight'].mean(), 0.0, places=5, msg="Mean of negative values after adjustment should be 0.")

if __name__ == '__main__':
    unittest.main()