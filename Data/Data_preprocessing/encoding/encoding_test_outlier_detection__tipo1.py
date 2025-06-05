import unittest
import pandas as pd
from outlier_detection import remove_outliers_iqr, remove_outliers_zscore

class TestOutlierDetection(unittest.TestCase):
    def setUp(self):
        self.data_frame = pd.DataFrame({
            'Salary': [50000, 60000, 55000, 120000, 58000, 59000, 61000]
        })

    def test_remove_outliers_iqr(self):
        cleaned_data_frame = remove_outliers_iqr(self.data_frame, 'Salary')
        
        self.assertFalse((cleaned_data_frame['Salary'] == 120000).any())
        
        self.assertEqual(len(cleaned_data_frame), len(self.data_frame) - 1, "One row should be removed when outlier is detected.")

    def test_remove_outliers_zscore(self):
        cleaned_data_frame = remove_outliers_zscore(self.data_frame, 'Salary', threshold=2)
        
        self.assertFalse((cleaned_data_frame['Salary'] == 120000).any())
        
        self.assertEqual(len(cleaned_data_frame), len(self.data_frame) - 1, "One row should be removed when outlier is detected.")

    def test_no_outliers(self):
        data_frame_no_outliers = pd.DataFrame({
            'Salary': [50000, 60000, 55000, 58000, 59000, 61000]
        })
        
        cleaned_data_frame = remove_outliers_iqr(data_frame_no_outliers, 'Salary')
        self.assertEqual(len(cleaned_data_frame), len(data_frame_no_outliers), "No rows should be removed if there are no outliers.")
        
        cleaned_data_frame = remove_outliers_zscore(data_frame_no_outliers, 'Salary', threshold=2)
        self.assertEqual(len(cleaned_data_frame), len(data_frame_no_outliers), "No rows should be removed if there are no outliers.")

    def test_multiple_outliers(self):
        data_frame_multiple_outliers = pd.DataFrame({
            'Salary': [50000, 60000, 55000, 120000, 130000, 140000, 61000]
        })
        
        cleaned_data_frame = remove_outliers_iqr(data_frame_multiple_outliers, 'Salary')
        self.assertNotIn(120000, cleaned_data_frame['Salary'].values)
        self.assertNotIn(130000, cleaned_data_frame['Salary'].values)
        self.assertNotIn(140000, cleaned_data_frame['Salary'].values)
        self.assertEqual(len(cleaned_data_frame), len(data_frame_multiple_outliers) - 3, "Three outliers should be removed.")
        
        cleaned_data_frame = remove_outliers_zscore(data_frame_multiple_outliers, 'Salary', threshold=2)
        self.assertNotIn(120000, cleaned_data_frame['Salary'].values)
        self.assertNotIn(130000, cleaned_data_frame['Salary'].values)
        self.assertNotIn(140000, cleaned_data_frame['Salary'].values)
        self.assertEqual(len(cleaned_data_frame), len(data_frame_multiple_outliers) - 3, "Three outliers should be removed.")

if __name__ == '__main__':
    unittest.main()