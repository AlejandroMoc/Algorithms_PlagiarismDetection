import unittest
from qbxz_complex_operations import add_qbxz, multiply_qbxz
from sqwerty import sqwerty
from fibo_seq import fibo_seq
from mtx_ops import add_matrices, multiply_matrices
from tringle_pascals import tringle_pascals
from vct_ops import add_vectors

class TestCalculation(unittest.TestCase):

    # Tests for Qbxz Complex Operations
    def test_add_qbxz(self):
        self.assertEqual(add_qbxz((1, 2), (3, 4)), (4, 6))

    def test_multiply_qbxz(self):
        self.assertEqual(multiply_qbxz((1, 2), (3, 4)), (-5, 10))

    def test_add_qbxz_invalid(self):
        with self.assertRaises(TypeError):
            add_qbxz((1, 2), "string")

    # Tests for Sqwerty
    def test_sqwerty_2x2(self):
        self.assertEqual(sqwerty([[1, 2], [3, 4]]), -2)

    def test_sqwerty_3x3(self):
        self.assertEqual(sqwerty([[1, 2, 3], [0, 1, 4], [5, 6, 0]]), 1)  # Corrected expected value

    def test_sqwerty_invalid(self):
        with self.assertRaises(ValueError):
            sqwerty([[1, 2], [3]])

    # Tests for Fibo Sequence
    def test_fibo_seq_count(self):
        self.assertEqual(fibo_seq(count=10), [0, 1, 1, 2, 3, 5, 8, 13, 21, 34])

    def test_fibo_seq_max_value(self):
        self.assertEqual(fibo_seq(max_value=20), [0, 1, 1, 2, 3, 5, 8, 13])

    def test_fibo_seq_invalid(self):
        with self.assertRaises(ValueError):
            fibo_seq()

    # Tests for Mtx Operations
    def test_add_matrices(self):
        self.assertEqual(add_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[6, 8], [10, 12]])

    def test_multiply_matrices(self):
        self.assertEqual(multiply_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[19, 22], [43, 50]])

    def test_add_matrices_invalid(self):
        with self.assertRaises(ValueError):
            add_matrices([[1, 2]], [[3, 4], [5, 6]])

    # Tests for Tringle Pascals
    def test_tringle_pascals(self):
        expected = [
            [1],
            [1, 1],
            [1, 2, 1],
            [1, 3, 3, 1],
            [1, 4, 6, 4, 1]
        ]
        self.assertEqual(tringle_pascals(5), expected)

    def test_tringle_pascals_invalid(self):
        with self.assertRaises(ValueError):
            tringle_pascals(0)

    # Tests for Vct Operations
    def test_add_vectors(self):
        self.assertEqual(add_vectors([1, 2, 3], [4, 5, 6]), [5, 7, 9])

    def test_add_vectors_invalid(self):
        with self.assertRaises(ValueError):
            add_vectors([1, 2], [1, 2, 3])

if __name__ == "__main__":
    unittest.main() 