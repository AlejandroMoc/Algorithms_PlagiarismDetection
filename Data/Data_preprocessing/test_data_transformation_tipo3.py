import unittest
import pandas as pd
import numpy as np
from data_transformation import log_transform, power_transform, binarize

class TestTransformData(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame({
            'Values': [100, 150, 200, 250, 300]
        })

    def test_log_transformation(self):
        transformed_df = log_transform(self.dataframe.copy(), 'Values')
        expected = np.log1p([100, 150, 200, 250, 300])
        pd.testing.assert_series_equal(transformed_df['Values'], pd.Series(expected), check_dtype=False)

    def test_power_transformation(self):
        transformed_df = power_transform(self.dataframe.copy(), 'Values', power=2)
        expected = [100**2, 150**2, 200**2, 250**2, 300**2]
        pd.testing.assert_series_equal(transformed_df['Values'], pd.Series(expected), check_dtype=False)

    def test_binarization(self):
        transformed_df = binarize(self.dataframe.copy(), 'Values', threshold=200)
        expected = [0, 0, 1, 1, 1]
        pd.testing.assert_series_equal(transformed_df['Values'], pd.Series(expected), check_dtype=False)

    def test_log_transformation_edge_case(self):
        df_zero = pd.DataFrame({'Values': [0, 1, 2]})
        transformed_df = log_transform(df_zero, 'Values')
        expected = np.log1p([0, 1, 2])
        pd.testing.assert_series_equal(transformed_df['Values'], pd.Series(expected), check_dtype=False)

    def test_power_transformation_negative_power(self):
        transformed_df = power_transform(self.dataframe.copy(), 'Values', power=-2)
        expected = [100**-2, 150**-2, 200**-2, 250**-2, 300**-2]
        pd.testing.assert_series_equal(transformed_df['Values'], pd.Series(expected), check_dtype=False)

    def test_binarization_edge_case(self):
        transformed_df = binarize(self.dataframe.copy(), 'Values', threshold=50)
        expected = [1, 1, 1, 1, 1]
        pd.testing.assert_series_equal(transformed_df['Values'], pd.Series(expected), check_dtype=False)

    def test_empty_data_frame(self):
        empty_df = pd.DataFrame({'Values': []})
        transformed_df = log_transform(empty_df, 'Values')
        self.assertTrue(transformed_df.empty, "The transformed DataFrame should be empty.")

        transformed_df = power_transform(empty_df, 'Values', power=2)
        self.assertTrue(transformed_df.empty, "The transformed DataFrame should be empty.")

        transformed_df = binarize(empty_df, 'Values', threshold=200)
        self.assertTrue(transformed_df.empty, "The transformed DataFrame should be empty.")

    def test_binarize_non_numeric_column(self):
        df_non_numeric = pd.DataFrame({'Values': ['a', 'b', 'c', 'd', 'e']})
        with self.assertRaises(TypeError):
            binarize(df_non_numeric, 'Values', threshold=200)

if __name__ == '__main__':
    unittest.main()