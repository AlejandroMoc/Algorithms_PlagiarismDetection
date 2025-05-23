import unittest
import pandas as pd
from outlier_detection import eliminate_outliers_iqr, eliminate_outliers_zscore

class TestOutlierDetection(unittest.TestCase):
    def setUp(self):
        # A sample DataFrame with salary data, including an outlier (120000)
        self.tbl = pd.DataFrame({
            'Earnings': [50000, 60000, 55000, 120000, 58000, 59000, 61000]
        })

    def test_eliminate_outliers_iqr(self):
        # Remove outliers using IQR method
        cleaned_tbl = eliminate_outliers_iqr(self.tbl, 'Earnings')
        
        # Ensure that the outlier (120000) is removed
        self.assertFalse((cleaned_tbl['Earnings'] == 120000).any())
        
        # Check if the length of the DataFrame is reduced by one (1 outlier should be removed)
        self.assertEqual(len(cleaned_tbl), len(self.tbl) - 1, "One row should be removed when outlier is detected.")

    def test_eliminate_outliers_zscore(self):
        # Remove outliers using Z-score method (with a threshold of 2)
        cleaned_tbl = eliminate_outliers_zscore(self.tbl, 'Earnings', threshold=2)
        
        # Ensure that the outlier (120000) is removed
        self.assertFalse((cleaned_tbl['Earnings'] == 120000).any())
        
        # Check if the length of the DataFrame is reduced by one (1 outlier should be removed)
        self.assertEqual(len(cleaned_tbl), len(self.tbl) - 1, "One row should be removed when outlier is detected.")

    def test_no_outliers(self):
        # A DataFrame with no outliers
        tbl_no_outliers = pd.DataFrame({
            'Earnings': [50000, 60000, 55000, 58000, 59000, 61000]
        })
        
        # Check if the length remains the same
        cleaned_tbl = eliminate_outliers_iqr(tbl_no_outliers, 'Earnings')
        self.assertEqual(len(cleaned_tbl), len(tbl_no_outliers), "No rows should be removed if there are no outliers.")
        
        # Also test using Z-score method with a threshold of 2
        cleaned_tbl = eliminate_outliers_zscore(tbl_no_outliers, 'Earnings', threshold=2)
        self.assertEqual(len(cleaned_tbl), len(tbl_no_outliers), "No rows should be removed if there are no outliers.")

    def test_multiple_outliers(self):
        # Data with multiple outliers
        tbl_multiple_outliers = pd.DataFrame({
            'Earnings': [50000, 60000, 55000, 120000, 130000, 140000, 61000]
        })
        
        # Remove outliers using IQR method
        cleaned_tbl = eliminate_outliers_iqr(tbl_multiple_outliers, 'Earnings')
        self.assertNotIn(120000, cleaned_tbl['Earnings'].values)
        self.assertNotIn(130000, cleaned_tbl['Earnings'].values)
        self.assertNotIn(140000, cleaned_tbl['Earnings'].values)
        self.assertEqual(len(cleaned_tbl), len(tbl_multiple_outliers) - 3, "Three outliers should be removed.")
        
        # Remove outliers using Z-score method
        cleaned_tbl = eliminate_outliers_zscore(tbl_multiple_outliers, 'Earnings', threshold=2)
        self.assertNotIn(120000, cleaned_tbl['Earnings'].values)
        self.assertNotIn(130000, cleaned_tbl['Earnings'].values)
        self.assertNotIn(140000, cleaned_tbl['Earnings'].values)
        self.assertEqual(len(cleaned_tbl), len(tbl_multiple_outliers) - 3, "Three outliers should be removed.")

if __name__ == '__main__':
    unittest.main()