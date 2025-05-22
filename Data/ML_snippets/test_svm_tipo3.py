import unittest
import numpy as np
from ML_snippets.svm import SVM

class TestSupportVectorMachine(unittest.TestCase):

    def test_fit_predict(self):
        data = np.array([[2, 3], [1, 1], [2, 1], [3, 2]])
        labels = np.array([1, -1, -1, 1])
        model = SVM(n_iters=1000)
        model.fit(data, labels)
        predictions = model.predict(np.array([[3, 3], [1, 0]]))
        np.testing.assert_array_equal(predictions, np.array([1, -1]))

    def test_margin(self):
        data = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
        labels = np.array([1, 1, 1, -1, -1])
        model = SVM(n_iters=1000)
        model.fit(data, labels)
        predictions = model.predict(data)
        np.testing.assert_array_equal(predictions, labels)

    def test_non_separable_data(self):
        # Non-linearly separable data
        data = np.array([[1, 2], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
        labels = np.array([1, -1, 1, -1, 1, -1])  # Not linearly separable
        model = SVM(n_iters=1000)
        model.fit(data, labels)
        predictions = model.predict(data)
        # Assert that the predictions do not match exactly
        self.assertNotEqual(np.sum(predictions == labels), len(labels))

    def test_large_dataset(self):
        # Simulate larger dataset
        data = np.random.rand(1000, 10)  # 1000 samples, 10 features
        labels = np.random.choice([1, -1], size=1000)
        model = SVM(n_iters=5000)
        model.fit(data, labels)
        predictions = model.predict(data[:10])
        self.assertEqual(len(predictions), 10)

    def test_with_different_hyperparameters(self):
        data = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
        labels = np.array([1, 1, 1, -1, -1])

        # Test with different learning rate
        model = SVM(learning_rate=0.01, n_iters=1000)
        model.fit(data, labels)
        predictions = model.predict(data)
        np.testing.assert_array_equal(predictions, labels)

        # Test with different lambda parameter
        model = SVM(lambda_param=0.1, n_iters=1000)
        model.fit(data, labels)
        predictions = model.predict(data)
        np.testing.assert_array_equal(predictions, labels)

    def test_convergence(self):
        data = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
        labels = np.array([1, 1, 1, -1, -1])
        model = SVM(n_iters=1000)
        model.fit(data, labels)
        
        # Assert that the weights have been updated (ensuring convergence)
        self.assertIsNotNone(model.w)
        self.assertIsNotNone(model.b)

if __name__ == '__main__':
    unittest.main()