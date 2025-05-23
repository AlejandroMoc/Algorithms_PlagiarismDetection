import unittest
import pandas as pd
import numpy as np
from missing_values import impute_numeric_with_mean, impute_categorical_with_mode, drop_missing

class TestMissingValues(unittest.TestCase):
    def setUp(self):
        self.data_frame = pd.DataFrame({
            'Age': [25, np.nan, 30, 22],
            'Gender': ['M', 'F', np.nan, 'F'],
            'Income': [50000, 60000, np.nan, 52000]
        })

    def test_impute_numeric_with_mean(self):
        imputed_data_frame = impute_numeric_with_mean(self.data_frame.copy(), ['Age', 'Income'])
        self.assertAlmostEqual(imputed_data_frame.loc[1, 'Age'], 25.666666666666668)
        self.assertAlmostEqual(imputed_data_frame.loc[2, 'Income'], 54000.0)

    def test_impute_categorical_with_mode(self):
        imputed_data_frame = impute_categorical_with_mode(self.data_frame.copy(), ['Gender'])
        self.assertEqual(imputed_data_frame.loc[2, 'Gender'], 'F')

    def test_drop_missing(self):
        cleaned_data_frame = drop_missing(self.data_frame.copy(), threshold=0.67)
        self.assertEqual(len(cleaned_data_frame), 3)

    def test_impute_numeric_with_mean_all_missing(self):
        data_frame_all_missing = pd.DataFrame({
            'Age': [25, np.nan, 30, 22],
            'Gender': ['M', 'F', np.nan, 'F'],
            'Income': [np.nan, np.nan, np.nan, np.nan]
        })
        imputed_data_frame = impute_numeric_with_mean(data_frame_all_missing, ['Income'])
        self.assertAlmostEqual(imputed_data_frame['Income'].iloc[0], np.nan)

    def test_drop_missing_empty_data_frame(self):
        empty_data_frame = pd.DataFrame(columns=['Age', 'Gender', 'Income'])
        cleaned_data_frame = drop_missing(empty_data_frame, threshold=0.5)
        self.assertTrue(cleaned_data_frame.empty, "Empty DataFrame should remain empty.")

    def test_impute_categorical_with_mode_empty(self):
        data_frame_empty_cat = pd.DataFrame({
            'Age': [25, np.nan, 30, 22],
            'Gender': [np.nan, np.nan, np.nan, np.nan],
            'Income': [50000, 60000, np.nan, 52000]
        })
        imputed_data_frame = impute_categorical_with_mode(data_frame_empty_cat, ['Gender'])
        self.assertEqual(imputed_data_frame['Gender'].isna().sum(), 0)

    def test_drop_missing_threshold_1(self):
        data_frame_with_missing = pd.DataFrame({
            'Age': [25, np.nan, 30, 22],
            'Gender': ['M', 'F', np.nan, 'F'],
            'Income': [50000, 60000, np.nan, 52000]
        })
        cleaned_data_frame = drop_missing(data_frame_with_missing, threshold=1)
        self.assertEqual(len(cleaned_data_frame), 0, "All rows should be dropped when threshold is 1.0.")

    def test_drop_missing_no_missing(self):
        data_frame_no_missing = pd.DataFrame({
            'Age': [25, 30, 30, 22],
            'Gender': ['M', 'F', 'M', 'F'],
            'Income': [50000, 60000, 65000, 52000]
        })
        cleaned_data_frame = drop_missing(data_frame_no_missing, threshold=0.67)
        self.assertEqual(len(cleaned_data_frame), 4, "No rows should be dropped when there are no missing values.")

if __name__ == '__main__':
    unittest.main()