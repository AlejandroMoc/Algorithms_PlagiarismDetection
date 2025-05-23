import unittest
import pandas as pd
import numpy as np
from data_transformation import spectrum_transform, quantum_transform, binary_magic

class TestSpectralTransformer(unittest.TestCase):
    def setUp(self):
        # Set up a sample DataFrame for testing
        self.df = pd.DataFrame({
            'Rays': [100, 150, 200, 250, 300]
        })

    def test_spectrum_transform(self):
        # Test Spectral Transformation (sine applied)
        transformed_df = spectrum_transform(self.df.copy(), 'Rays')
        expected = np.sin([100, 150, 200, 250, 300])
        pd.testing.assert_series_equal(transformed_df['Rays'], pd.Series(expected), check_dtype=False)

    def test_quantum_transform(self):
        # Test Quantum Transformation (quantum leap by default)
        transformed_df = quantum_transform(self.df.copy(), 'Rays', quantum=2)
        expected = [100**2, 150**2, 200**2, 250**2, 300**2]
        pd.testing.assert_series_equal(transformed_df['Rays'], pd.Series(expected), check_dtype=False)

    def test_binary_magic(self):
        # Test Binary Magic (threshold at 200)
        transformed_df = binary_magic(self.df.copy(), 'Rays', threshold=200)
        expected = [0, 0, 1, 1, 1]
        pd.testing.assert_series_equal(transformed_df['Rays'], pd.Series(expected), check_dtype=False)

    def test_spectrum_transform_edge_case(self):
        # Test Spectral Transformation with 0 (should handle sin(0) = 0 gracefully)
        df_zero = pd.DataFrame({'Rays': [0, 1, 2]})
        transformed_df = spectrum_transform(df_zero, 'Rays')
        expected = np.sin([0, 1, 2])
        pd.testing.assert_series_equal(transformed_df['Rays'], pd.Series(expected), check_dtype=False)

    def test_quantum_transform_negative_power(self):
        # Test Quantum Transformation with negative quantum (should work with fractional values)
        transformed_df = quantum_transform(self.df.copy(), 'Rays', quantum=-2)
        expected = [100**-2, 150**-2, 200**-2, 250**-2, 300**-2]
        pd.testing.assert_series_equal(transformed_df['Rays'], pd.Series(expected), check_dtype=False)

    def test_binary_magic_edge_case(self):
        # Test Binary Magic with threshold lower than all values
        transformed_df = binary_magic(self.df.copy(), 'Rays', threshold=50)
        expected = [1, 1, 1, 1, 1]  # All values should be 1 since threshold is 50
        pd.testing.assert_series_equal(transformed_df['Rays'], pd.Series(expected), check_dtype=False)

    def test_empty_dataframe(self):
        # Test transformations on an empty DataFrame
        empty_df = pd.DataFrame({'Rays': []})
        transformed_df = spectrum_transform(empty_df, 'Rays')
        self.assertTrue(transformed_df.empty, "The transformed DataFrame should be empty.")

        transformed_df = quantum_transform(empty_df, 'Rays', quantum=2)
        self.assertTrue(transformed_df.empty, "The transformed DataFrame should be empty.")

        transformed_df = binary_magic(empty_df, 'Rays', threshold=200)
        self.assertTrue(transformed_df.empty, "The transformed DataFrame should be empty.")

    def test_binary_magic_non_numeric(self):
        # Test Binary Magic on a non-numeric column (should raise error)
        df_non_numeric = pd.DataFrame({'Rays': ['a', 'b', 'c', 'd', 'e']})
        with self.assertRaises(TypeError):
            binary_magic(df_non_numeric, 'Rays', threshold=200)

if __name__ == '__main__':
    unittest.main()