import unittest
import numpy as np
from advanced_graphing import (
    adjust_configuration, standardize_values, calculate_moving_average, secure_display, line_graph, bar_graph, fetch_styles, configuration
)

class TestAdvancedGraphing(unittest.TestCase):

    def setUp(self):
        """Prepare the testing environment."""
        adjust_configuration(style='ggplot', figsize=(10, 6), color_palette='viridis')

    def test_standardize_values(self):
        """Verify the standardization of values."""
        data = np.array([1, 2, 3, 4, 5])
        standardized = standardize_values(data)
        expected = np.array([0, 0.25, 0.5, 0.75, 1])
        np.testing.assert_array_almost_equal(standardized, expected)

    def test_calculate_moving_average(self):
        """Check the moving average computation."""
        data = np.array([1, 2, 3, 4, 5])
        moving_avg = calculate_moving_average(data, window=3)
        expected = np.array([2, 3, 4])
        np.testing.assert_array_almost_equal(moving_avg, expected)

    def test_adjust_configuration(self):
        """Test adjustment of configuration."""
        available_styles = fetch_styles()
        chosen_style = available_styles[0]  # Use the first available style
        adjust_configuration(style=chosen_style, figsize=(12, 8), color_palette='Set1')
        
        # Access the global configuration directly
        self.assertEqual(configuration.style, chosen_style)
        self.assertEqual(configuration.figsize, (12, 8))
        self.assertEqual(configuration.color_palette, 'Set1')

    def test_secure_display(self):
        """Ensure secure_display doesn't raise an error for valid functions."""
        try:
            secure_display(line_graph, np.array([1, 2, 3]), np.array([1, 4, 9]), title="Test Line Graph")
            secure_display(bar_graph, ['A', 'B', 'C'], [1, 2, 3], title="Test Bar Graph")
        except Exception as e:
            self.fail(f"secure_display raised an exception: {str(e)}")

if __name__ == '__main__':
    unittest.main()