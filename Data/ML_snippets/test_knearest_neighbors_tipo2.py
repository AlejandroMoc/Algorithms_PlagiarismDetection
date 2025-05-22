import unittest
import numpy as np
from ML_snippets.knearest_neighbors import KNearestNeighbors

class TestKNN(unittest.TestCase):
    
    def setUp(self):
        # Setup common training data
        self.features = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 7], [8, 6]])
        self.labels = np.array([0, 0, 0, 1, 1, 1])
    
    def test_fit_predict_k3(self):
        knn = KNearestNeighbors(k=3)
        knn.train(self.features, self.labels)
        preds = knn.infer(np.array([[5, 5], [2, 2]]))
        np.testing.assert_array_equal(preds, np.array([1, 0]))

    def test_k_one(self):
        knn = KNearestNeighbors(k=1)
        knn.train(self.features, self.labels)
        preds = knn.infer(np.array([[1.5]]))
        np.testing.assert_array_equal(preds, np.array([1]))

    def test_k_large(self):
        knn = KNearestNeighbors(k=10)  # k greater than the number of samples
        knn.train(self.features, self.labels)
        preds = knn.infer(np.array([[1, 1], [6, 6]]))
        np.testing.assert_array_equal(preds, np.array([0, 1]))

    def test_predict_multi_classes(self):
        features = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        labels = np.array([0, 0, 1, 1, 2])
        knn = KNearestNeighbors(k=3)
        knn.train(features, labels)
        preds = knn.infer(np.array([[2.5, 2.5], [3.5, 3.5]]))
        np.testing.assert_array_equal(preds, np.array([0, 1]))

    def test_same_label_case(self):
        features = np.array([[1, 2], [2, 3], [3, 3]])
        labels = np.array([1, 1, 1])  # All the same label
        knn = KNearestNeighbors(k=2)
        knn.train(features, labels)
        preds = knn.infer(np.array([[1, 1]]))
        np.testing.assert_array_equal(preds, np.array([1]))

    def test_invalid_k_value(self):
        knn = KNearestNeighbors(k=10)
        knn.train(self.features, self.labels)
        # k is larger than the number of samples, so it should predict based on all points
        preds = knn.infer(np.array([[1, 1], [5, 5]]))
        np.testing.assert_array_equal(preds, np.array([0, 1]))

if __name__ == '__main__':
    unittest.main()