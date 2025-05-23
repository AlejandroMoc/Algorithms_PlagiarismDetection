import unittest
import pandas as pd
from encoding import funky_encode, groovy_encode

class TestFunkyGroovy(unittest.TestCase):
    def setUp(self):
        # Set up a sample DataFrame for testing
        self.data_frame = pd.DataFrame({
            'Style': ['Bold', 'Italic', 'Regular', 'Italic']
        })

    def test_funky_encode(self):
        # Test Funky Encoding
        encoded_data_frame = funky_encode(self.data_frame, 'Style')
        self.assertIn('Style_Regular', encoded_data_frame.columns, "Encoded DataFrame should contain 'Style_Regular'.")
        self.assertIn('Style_Bold', encoded_data_frame.columns, "Encoded DataFrame should contain 'Style_Bold'.")
        self.assertNotIn('Style', encoded_data_frame.columns, "Original 'Style' column should be dropped.")
        self.assertEqual(encoded_data_frame.shape[1], 3, "The number of columns after funky encoding is incorrect.")

    def test_groovy_encode(self):
        # Test Groovy Encoding
        encoded_data_frame = groovy_encode(self.data_frame.copy(), 'Style')
        self.assertTrue(encoded_data_frame['Style'].dtype == int, "Encoded column should have integer dtype.")
        self.assertEqual(encoded_data_frame['Style'].min(), 0, "The minimum label value should be 0.")
        self.assertEqual(encoded_data_frame['Style'].max(), 2, "The maximum label value should be 2 (for 'Bold', 'Italic', 'Regular').")
        
    def test_funky_encode_empty_df(self):
        # Test Funky Encoding with an empty DataFrame
        empty_data_frame = pd.DataFrame(columns=['Style'])
        encoded_data_frame = funky_encode(empty_data_frame, 'Style')
        self.assertTrue(encoded_data_frame.empty, "The encoded DataFrame should be empty when the input is empty.")

    def test_groovy_encode_empty_df(self):
        # Test Groovy Encoding with an empty DataFrame
        empty_data_frame = pd.DataFrame(columns=['Style'])
        encoded_data_frame = groovy_encode(empty_data_frame, 'Style')
        self.assertTrue(encoded_data_frame.empty, "The encoded DataFrame should be empty when the input is empty.")
    
    def test_funky_encode_single_unique_value(self):
        # Test Funky Encoding when there is only one unique value in the column
        single_value_data_frame = pd.DataFrame({'Style': ['Bold', 'Bold', 'Bold', 'Bold']})
        encoded_data_frame = funky_encode(single_value_data_frame, 'Style')
        self.assertIn('Style_Bold', encoded_data_frame.columns, "Encoded DataFrame should contain 'Style_Bold'.")
        self.assertEqual(encoded_data_frame.shape[1], 1, "The number of columns after funky encoding with a single value should be 1.")
    
    def test_groovy_encode_single_unique_value(self):
        # Test Groovy Encoding when there is only one unique value in the column
        single_value_data_frame = pd.DataFrame({'Style': ['Bold', 'Bold', 'Bold', 'Bold']})
        encoded_data_frame = groovy_encode(single_value_data_frame, 'Style')
        self.assertEqual(encoded_data_frame['Style'].min(), 0, "The encoded value should be 0 for a single unique value.")
        self.assertEqual(encoded_data_frame['Style'].max(), 0, "The encoded value should be 0 for a single unique value.")

    def test_groovy_encode_non_string_column(self):
        # Test Groovy Encoding with a non-string column (e.g., numeric)
        numeric_data_frame = pd.DataFrame({'Style': [1, 2, 1, 3]})
        encoded_data_frame = groovy_encode(numeric_data_frame, 'Style')
        self.assertTrue(encoded_data_frame['Style'].dtype == int, "Encoded column should have integer dtype.")
        self.assertEqual(encoded_data_frame['Style'].min(), 0, "The minimum label value should be 0.")

if __name__ == '__main__':
    unittest.main()