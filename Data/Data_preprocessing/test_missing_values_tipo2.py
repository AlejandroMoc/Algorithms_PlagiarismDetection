import unittest
import pandas as pd
import numpy as np
from missing_values import process_numeric_with_mean, process_categorical_with_mode, eliminate_missing

class TestMystery(unittest.TestCase):
    def setUp(self):
        self.data_frame = pd.DataFrame({
            'X1': [25, np.nan, 30, 22],
            'X2': ['M', 'F', np.nan, 'F'],
            'X3': [50000, 60000, np.nan, 52000]
        })

    def test_process_numeric_with_mean(self):
        processed_df = process_numeric_with_mean(self.data_frame.copy(), ['X1', 'X3'])
        self.assertAlmostEqual(processed_df.loc[1, 'X1'], 25.666666666666668)
        self.assertAlmostEqual(processed_df.loc[2, 'X3'], 54000.0)

    def test_process_categorical_with_mode(self):
        processed_df = process_categorical_with_mode(self.data_frame.copy(), ['X2'])
        self.assertEqual(processed_df.loc[2, 'X2'], 'F')

    def test_eliminate_missing(self):
        cleaned_df = eliminate_missing(self.data_frame.copy(), threshold=0.67)
        self.assertEqual(len(cleaned_df), 3)

    def test_process_numeric_with_mean_all_missing(self):
        # All values are missing in the column 'X3', check processing with mean.
        df_all_missing = pd.DataFrame({
            'X1': [25, np.nan, 30, 22],
            'X2': ['M', 'F', np.nan, 'F'],
            'X3': [np.nan, np.nan, np.nan, np.nan]
        })
        processed_df = process_numeric_with_mean(df_all_missing, ['X3'])
        self.assertAlmostEqual(processed_df['X3'].iloc[0], np.nan)

    def test_eliminate_missing_empty_df(self):
        # Test for an empty DataFrame
        empty_df = pd.DataFrame(columns=['X1', 'X2', 'X3'])
        cleaned_df = eliminate_missing(empty_df, threshold=0.5)
        self.assertTrue(cleaned_df.empty, "Empty DataFrame should remain empty.")

    def test_process_categorical_with_mode_empty(self):
        # Test process categorical with mode for empty column 'X2'
        df_empty_cat = pd.DataFrame({
            'X1': [25, np.nan, 30, 22],
            'X2': [np.nan, np.nan, np.nan, np.nan],
            'X3': [50000, 60000, np.nan, 52000]
        })
        processed_df = process_categorical_with_mode(df_empty_cat, ['X2'])
        self.assertEqual(processed_df['X2'].isna().sum(), 0)

    def test_eliminate_missing_threshold_1(self):
        # Test with a threshold of 1 (i.e., drop rows with any missing values)
        df_with_missing = pd.DataFrame({
            'X1': [25, np.nan, 30, 22],
            'X2': ['M', 'F', np.nan, 'F'],
            'X3': [50000, 60000, np.nan, 52000]
        })
        cleaned_df = eliminate_missing(df_with_missing, threshold=1)
        self.assertEqual(len(cleaned_df), 0, "All rows should be dropped when threshold is 1.0.")

    def test_eliminate_missing_no_missing(self):
        # Test for no missing values in the DataFrame
        df_no_missing = pd.DataFrame({
            'X1': [25, 30, 30, 22],
            'X2': ['M', 'F', 'M', 'F'],
            'X3': [50000, 60000, 65000, 52000]
        })
        cleaned_df = eliminate_missing(df_no_missing, threshold=0.67)
        self.assertEqual(len(cleaned_df), 4, "No rows should be dropped when there are no missing values.")

if __name__ == '__main__':
    unittest.main()