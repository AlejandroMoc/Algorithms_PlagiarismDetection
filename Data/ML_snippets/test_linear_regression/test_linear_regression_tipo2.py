import unittest
import numpy as np
from ML_snippets.linear_regression import LinearRegressionModel

class TestLinearRegressionModel(unittest.TestCase):
    
    def setUp(self):
        # Setup common training data
        self.X_mystic = np.array([[1], [2], [3], [4], [5]])
        self.y_mystic = np.array([2, 4, 6, 8, 10])
        self.mysterious_model = LinearRegressionModel()

    def test_divine_foresight(self):
        self.mysterious_model.fit(self.X_mystic, self.y_mystic)
        predictions = self.mysterious_model.predict(np.array([[6], [7]]))
        np.testing.assert_array_almost_equal(predictions, np.array([12., 14.]))

    def test_unsolvable_puzzle(self):
        X_puzzle = np.array([[1, 1], [1, 1], [1, 1]])  # Singular matrix
        y_puzzle = np.array([2, 2, 2])
        with self.assertRaises(np.linalg.LinAlgError):
            self.mysterious_model.fit(X_puzzle, y_puzzle)

    def test_mysterious_fate(self):
        X_mysterious = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_mysterious = np.array([2, 4, 6, 8, 10])
        self.mysterious_model.fit(X_mysterious, y_mysterious)
        predictions = self.mysterious_model.predict(np.array([[6, 7], [7, 8]]))
        np.testing.assert_array_almost_equal(predictions, np.array([12., 14.]))

    def test_secret_encryption(self):
        X_secret = np.array([[1]])
        y_secret = np.array([2])
        self.mysterious_model.fit(X_secret, y_secret)
        prediction = self.mysterious_model.predict(np.array([[2]]))
        np.testing.assert_array_almost_equal(prediction, np.array([4.]))

    def test_unbreakable_code(self):
        X_unbreakable = np.array([[1], [2], [3], [4], [5]])
        y_unbreakable = np.array([2, 4, 6, 8, 10])
        self.mysterious_model.fit(X_unbreakable, y_unbreakable)
        predictions = self.mysterious_model.predict(X_unbreakable)
        np.testing.assert_array_almost_equal(predictions, y_unbreakable)

    def test_hidden_truth(self):
        X_hidden = np.array([[1], [2], [3], [4], [5]])
        y_hidden = np.array([1, 4, 9, 16, 25])  # Quadratic relationship
        self.mysterious_model.fit(X_hidden, y_hidden)
        predictions = self.mysterious_model.predict(np.array([[6]]))
        # Linear regression should not fit this well
        self.assertNotAlmostEqual(predictions[0], 36, places=2)

if __name__ == '__main__':
    unittest.main()