import unittest
import pandas as pd
from data_cleaning import remove_duplicates, replace_missing_with_mean, standardize_text

class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        # Set up a sample DataFrame for testing
        self.data_frame = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Alice', 'Charlie'],
            'Age': [25, 30, 25, None],
            'City': ['New York', 'Los Angeles', 'New York', 'Chicago']
        })

    def test_remove_duplicates(self):
        # Test that duplicates are removed
        cleaned_data_frame = remove_duplicates(self.data_frame)
        self.assertEqual(len(cleaned_data_frame), 3, "Duplicates were not removed correctly.")
        self.assertTrue('Alice' in cleaned_data_frame['Name'].values, "Duplicate for 'Alice' was not removed.")

    def test_replace_missing_with_mean(self):
        # Test replacing missing values with the mean
        cleaned_data_frame = replace_missing_with_mean(self.data_frame, 'Age')
        expected_mean = self.data_frame['Age'].mean()
        self.assertEqual(cleaned_data_frame.loc[3, 'Age'], expected_mean, "Missing value was not replaced correctly with the mean.")
        
    def test_standardize_text(self):
        # Test standardizing text (lowercase and stripped)
        standardized_data_frame = standardize_text(self.data_frame, 'Name')
        self.assertTrue(standardized_data_frame['Name'].str.islower().all(), "Text was not standardized to lowercase.")
        self.assertFalse(standardized_data_frame['Name'].str.contains(' ').any(), "Text was not stripped of whitespace.")

    def test_replace_missing_with_mean_no_missing_values(self):
        # Test replacing missing values when there are no missing values
        data_frame_no_missing = self.data_frame.copy()
        data_frame_no_missing['Age'].iloc[3] = 29  # No missing value in 'Age'
        cleaned_data_frame = replace_missing_with_mean(data_frame_no_missing, 'Age')
        self.assertEqual(cleaned_data_frame.loc[3, 'Age'], 29, "The value should remain unchanged when there is no missing value.")

    def test_remove_duplicates_empty_df(self):
        # Test that no error is raised when removing duplicates from an empty DataFrame
        empty_data_frame = pd.DataFrame(columns=self.data_frame.columns)
        cleaned_data_frame = remove_duplicates(empty_data_frame)
        self.assertEqual(len(cleaned_data_frame), 0, "Removing duplicates from an empty DataFrame failed.")

    def test_standardize_text_empty_column(self):
        # Test standardizing text for an empty column
        empty_data_frame = pd.DataFrame({'Name': []})
        cleaned_data_frame = standardize_text(empty_data_frame, 'Name')
        self.assertEqual(len(cleaned_data_frame), 0, "Standardizing text on an empty column failed.")

if __name__ == '__main__':
    unittest.main()