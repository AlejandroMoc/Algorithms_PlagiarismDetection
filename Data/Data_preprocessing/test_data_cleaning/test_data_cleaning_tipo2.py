import unittest
import pandas as pd
from data_cleaning import purify_data, fill_empty_with_average, cleanse_words

class TestPurification(unittest.TestCase):
    def setUp(self):
        # Set up a sample DataFrame for testing
        self.data_table = pd.DataFrame({
            'Identity': ['Alpha', 'Beta', 'Alpha', 'Delta'],
            'Years': [25, 30, 25, None],
            'Location': ['Tokyo', 'Paris', 'Tokyo', 'Sydney']
        })

    def test_purify_data(self):
        # Test that duplicates are removed
        purified_table = purify_data(self.data_table)
        self.assertEqual(len(purified_table), 3, "Duplicates were not removed correctly.")
        self.assertTrue('Alpha' in purified_table['Identity'].values, "Duplicate for 'Alpha' was not removed.")

    def test_fill_empty_with_average(self):
        # Test replacing missing values with the mean
        purified_table = fill_empty_with_average(self.data_table, 'Years')
        expected_average = self.data_table['Years'].mean()
        self.assertEqual(purified_table.loc[3, 'Years'], expected_average, "Missing value was not replaced correctly with the mean.")
        
    def test_cleanse_words(self):
        # Test standardizing text (lowercase and stripped)
        cleansed_table = cleanse_words(self.data_table, 'Identity')
        self.assertTrue(cleansed_table['Identity'].str.islower().all(), "Text was not standardized to lowercase.")
        self.assertFalse(cleansed_table['Identity'].str.contains(' ').any(), "Text was not stripped of whitespace.")

    def test_fill_empty_with_average_no_empty_values(self):
        # Test replacing missing values when there are no missing values
        table_no_empty = self.data_table.copy()
        table_no_empty['Years'].iloc[3] = 29  # No missing value in 'Years'
        purified_table = fill_empty_with_average(table_no_empty, 'Years')
        self.assertEqual(purified_table.loc[3, 'Years'], 29, "The value should remain unchanged when there is no missing value.")

    def test_purify_data_empty_table(self):
        # Test that no error is raised when removing duplicates from an empty DataFrame
        empty_table = pd.DataFrame(columns=self.data_table.columns)
        purified_table = purify_data(empty_table)
        self.assertEqual(len(purified_table), 0, "Removing duplicates from an empty DataFrame failed.")

    def test_cleanse_words_empty_field(self):
        # Test standardizing text for an empty column
        empty_table = pd.DataFrame({'Identity': []})
        purified_table = cleanse_words(empty_table, 'Identity')
        self.assertEqual(len(purified_table), 0, "Standardizing text on an empty column failed.")

if __name__ == '__main__':
    unittest.main()