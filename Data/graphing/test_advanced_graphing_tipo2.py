import unittest
import numpy as np
from advanced_graphing import (
    set_customization, adjust_values, smooth_data, secure_display, curve_display, histogram_display, get_options, customization_params
)

class TestVisualizations(unittest.TestCase):

    def setUp(self):
        """Initialize the test environment."""
        set_customization(style='seaborn', dimensions=(10, 6), color_scheme='plasma')

    def test_adjust_values(self):
        """Verify the adjustment of values."""
        dataset = np.array([1, 2, 3, 4, 5])
        adjusted = adjust_values(dataset)
        expected = np.array([0, 0.25, 0.5, 0.75, 1])
        np.testing.assert_array_almost_equal(adjusted, expected)

    def test_smooth_data(self):
        """Confirm the smoothing of data."""
        dataset = np.array([1, 2, 3, 4, 5])
        smoothed = smooth_data(dataset, window=3)
        expected = np.array([2, 3, 4])
        np.testing.assert_array_almost_equal(smoothed, expected)

    def test_set_customization(self):
        """Test the customization settings."""
        available_options = get_options()
        chosen_option = available_options[0]  # Using the first available option
        set_customization(style=chosen_option, dimensions=(12, 8), color_scheme='Accent')
        
        # Directly check the global customization parameters
        self.assertEqual(customization_params.style, chosen_option)
        self.assertEqual(customization_params.dimensions, (12, 8))
        self.assertEqual(customization_params.color_scheme, 'Accent')

    def test_secure_display(self):
        """Ensure safe_display functions without error for valid inputs."""
        try:
            secure_display(curve_display, np.array([1, 2, 3]), np.array([1, 4, 9]), title="Test Curve Display")
            secure_display(histogram_display, ['A', 'B', 'C'], [1, 2, 3], title="Test Histogram Display")
        except Exception as e:
            self.fail(f"secure_display raised an exception: {str(e)}")

if __name__ == '__main__':
    unittest.main()