import unittest
import numpy as np
from ML_snippets.decision_tree import DecisionTreeClassifier

class TestRenamedClassifier(unittest.TestCase):
    
    def setUp(self):
        self.X_sample = np.array([[2], [3], [10], [19]])
        self.y_sample = np.array([0, 0, 1, 1])
        self.X_complex = np.array([[2, 3], [3, 3], [10, 12], [19, 20], [5, 5], [7, 8]])
        self.y_complex = np.array([0, 0, 1, 1, 0, 1])
    
    def test_basic_prediction(self):
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(self.X_sample, self.y_sample)
        predictions = clf.predict(np.array([[4], [15]]))
        np.testing.assert_array_equal(predictions, np.array([0, 1]))

    def test_single_class_prediction(self):
        X = np.array([[1], [2], [3]])
        y = np.array([1, 1, 1])
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        predictions = clf.predict(np.array([[4], [5]]))
        np.testing.assert_array_equal(predictions, np.array([1, 1]))
    
    def test_complex_features(self):
        clf = DecisionTreeClassifier(max_depth=2)
        clf.fit(self.X_complex, self.y_complex)
        predictions = clf.predict(np.array([[6, 7], [18, 19]]))
        self.assertEqual(len(predictions), 2)

    def test_unlimited_depth_tree(self):
        clf = DecisionTreeClassifier(max_depth=None)
        clf.fit(self.X_sample, self.y_sample)
        predictions = clf.predict(np.array([[4], [15]]))
        np.testing.assert_array_equal(predictions, np.array([0, 1]))

    def test_invalid_data_input(self):
        X_invalid = np.array([[1], [2]])
        y_invalid = np.array([1])
        clf = DecisionTreeClassifier()
        with self.assertRaises(ValueError):
            clf.fit(X_invalid, y_invalid)

    def test_edge_case_single_sample(self):
        X = np.array([[1]])
        y = np.array([0])
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(X, y)
        prediction = clf.predict(np.array([[1]]))
        np.testing.assert_array_equal(prediction, np.array([0]))

if __name__ == '__main__':
    unittest.main()