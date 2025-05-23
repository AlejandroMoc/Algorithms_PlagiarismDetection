import unittest
import numpy as np
from ML_snippets.knearest_neighbors import KNearestNeighbors

class TestKNearestNeighbors(unittest.TestCase):
    
    def setUp(self):
        # Setup common training data
        self.X_train_data = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 7], [8, 6]])
        self.y_train_data = np.array([0, 0, 0, 1, 1, 1])
    
    def test_fit_predict_k3(self):
        knn_instance = KNearestNeighbors(k=3)
        knn_instance.fit(self.X_train_data, self.y_train_data)
        predictions = knn_instance.predict(np.array([[5, 5], [2, 2]]))
        np.testing.assert_array_equal(predictions, np.array([1, 0]))

    def test_k_equals_one(self):
        knn_instance = KNearestNeighbors(k=1)
        knn_instance.fit(self.X_train_data, self.y_train_data)
        predictions = knn_instance.predict(np.array([[1.5]]))
        np.testing.assert_array_equal(predictions, np.array([1]))

    def test_k_equals_large_value(self):
        knn_instance = KNearestNeighbors(k=10)  # k greater than the number of samples
        knn_instance.fit(self.X_train_data, self.y_train_data)
        predictions = knn_instance.predict(np.array([[1, 1], [6, 6]]))
        # Expect predictions based on majority voting, considering all samples
        np.testing.assert_array_equal(predictions, np.array([0, 1]))

    def test_predict_multiple_classes(self):
        X_train_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        y_train_data = np.array([0, 0, 1, 1, 2])
        knn_instance = KNearestNeighbors(k=3)
        knn_instance.fit(X_train_data, y_train_data)
        predictions = knn_instance.predict(np.array([[2.5, 2.5], [3.5, 3.5]]))
        # The first sample is closer to [2, 2] and [3, 3] (class 0 and 1), while the second is closer to [4, 4] and [5, 5] (class 1)
        np.testing.assert_array_equal(predictions, np.array([0, 1]))

    def test_edge_case_all_same_label(self):
        X_train_data = np.array([[1, 2], [2, 3], [3, 3]])
        y_train_data = np.array([1, 1, 1])  # All the same label
        knn_instance = KNearestNeighbors(k=2)
        knn_instance.fit(X_train_data, y_train_data)
        predictions = knn_instance.predict(np.array([[1, 1]]))
        np.testing.assert_array_equal(predictions, np.array([1]))

    def test_invalid_k(self):
        knn_instance = KNearestNeighbors(k=10)
        knn_instance.fit(self.X_train_data, self.y_train_data)
        # k is larger than the number of samples, so it should predict based on all points
        predictions = knn_instance.predict(np.array([[1, 1], [5, 5]]))
        np.testing.assert_array_equal(predictions, np.array([0, 1]))

if __name__ == '__main__':
    unittest.main()