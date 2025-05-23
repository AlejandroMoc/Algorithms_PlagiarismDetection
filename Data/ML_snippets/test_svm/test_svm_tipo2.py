import unittest
import numpy as np
from ML_snippets.svm import SVM

class TestSVM(unittest.TestCase):

    def test_fit_predict(self):
        A = np.array([[2, 3], [1, 1], [2, 1], [3, 2]])
        B = np.array([1, -1, -1, 1])
        model = SVMC(n_iters=1000)
        model.train(A, B)
        C = model.infer(np.array([[3, 3], [1, 0]]))
        np.testing.assert_array_equal(C, np.array([1, -1]))

    def test_margin(self):
        A = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
        B = np.array([1, 1, 1, -1, -1])
        model = SVMC(n_iters=1000)
        model.train(A, B)
        C = model.infer(A)
        np.testing.assert_array_equal(C, B)

    def test_non_separable_data(self):
        # Non-linearly separable data
        A = np.array([[1, 2], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
        B = np.array([1, -1, 1, -1, 1, -1])  # Not linearly separable
        model = SVMC(n_iters=1000)
        model.train(A, B)
        C = model.infer(A)
        # Assert that the predictions do not match exactly
        self.assertNotEqual(np.sum(C == B), len(B))

    def test_large_dataset(self):
        # Simulate larger dataset
        A = np.random.rand(1000, 10)  # 1000 samples, 10 features
        B = np.random.choice([1, -1], size=1000)
        model = SVMC(n_iters=5000)
        model.train(A, B)
        C = model.infer(A[:10])
        self.assertEqual(len(C), 10)

    def test_with_different_hyperparameters(self):
        A = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
        B = np.array([1, 1, 1, -1, -1])

        # Test with different learning rate
        model = SVMC(learning_rate=0.01, n_iters=1000)
        model.train(A, B)
        C = model.infer(A)
        np.testing.assert_array_equal(C, B)

        # Test with different lambda parameter
        model = SVMC(lambda_param=0.1, n_iters=1000)
        model.train(A, B)
        C = model.infer(A)
        np.testing.assert_array_equal(C, B)

    def test_convergence(self):
        A = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
        B = np.array([1, 1, 1, -1, -1])
        model = SVMC(n_iters=1000)
        model.train(A, B)
        
        # Assert that the weights have been updated (ensuring convergence)
        self.assertIsNotNone(model.w)
        self.assertIsNotNone(model.b)

if __name__ == '__main__':
    unittest.main()