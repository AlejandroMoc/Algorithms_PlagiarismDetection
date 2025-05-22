import unittest
import numpy as np
from ML_snippets.linear_regression import LinearRegressionModel

class TestLinearRegressionModel(unittest.TestCase):
    
    def setUp(self):
        # Setup common training data
        self.X_data = np.array([[1], [2], [3], [4], [5]])
        self.y_data = np.array([2, 4, 6, 8, 10])
        self.model_instance = LinearRegressionModel()

    def test_fit_predict(self):
        self.model_instance.fit(self.X_data, self.y_data)
        predictions = self.model_instance.predict(np.array([[6], [7]]))
        np.testing.assert_array_almost_equal(predictions, np.array([12., 14.]))

    def test_non_invertible_matrix(self):
        X_matrix = np.array([[1, 1], [1, 1], [1, 1]])  # Singular matrix
        y_values = np.array([2, 2, 2])
        with self.assertRaises(np.linalg.LinAlgError):
            self.model_instance.fit(X_matrix, y_values)

    def test_multivariate_fit_predict(self):
        X_multivariate = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_multivariate = np.array([2, 4, 6, 8, 10])
        self.model_instance.fit(X_multivariate, y_multivariate)
        predictions = self.model_instance.predict(np.array([[6, 7], [7, 8]]))
        np.testing.assert_array_almost_equal(predictions, np.array([12., 14.]))

    def test_single_sample(self):
        X_single = np.array([[1]])
        y_single = np.array([2])
        self.model_instance.fit(X_single, y_single)
        prediction = self.model_instance.predict(np.array([[2]]))
        np.testing.assert_array_almost_equal(prediction, np.array([4.]))

    def test_perfect_fit(self):
        X_perfect = np.array([[1], [2], [3], [4], [5]])
        y_perfect = np.array([2, 4, 6, 8, 10])
        self.model_instance.fit(X_perfect, y_perfect)
        predictions = self.model_instance.predict(X_perfect)
        np.testing.assert_array_almost_equal(predictions, y_perfect)

    def test_non_linear_data(self):
        X_non_linear = np.array([[1], [2], [3], [4], [5]])
        y_non_linear = np.array([1, 4, 9, 16, 25])  # Quadratic relationship
        self.model_instance.fit(X_non_linear, y_non_linear)
        predictions = self.model_instance.predict(np.array([[6]]))
        # Linear regression should not fit this well
        self.assertNotAlmostEqual(predictions[0], 36, places=2)

if __name__ == '__main__':
    unittest.main()