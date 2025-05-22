import unittest
from complex_number_operations import sum_complex, product_complex
from determinant import calculate_determinant
from fibonacci_sequence import calculate_fibonacci_sequence
from matrix_operations import sum_matrices, product_matrices
from pascals_triangle import generate_pascals_triangle
from vector_operations import sum_vectors

class TestMathematics(unittest.TestCase):

    # Tests for Complex Number Operations
    def test_sum_complex(self):
        self.assertEqual(sum_complex((1, 2), (3, 4)), (4, 6))

    def test_product_complex(self):
        self.assertEqual(product_complex((1, 2), (3, 4)), (-5, 10))

    def test_sum_complex_invalid(self):
        with self.assertRaises(TypeError):
            sum_complex((1, 2), "string")

    # Tests for Determinant
    def test_calculate_determinant_2x2(self):
        self.assertEqual(calculate_determinant([[1, 2], [3, 4]]), -2)

    def test_calculate_determinant_3x3(self):
        self.assertEqual(calculate_determinant([[1, 2, 3], [0, 1, 4], [5, 6, 0]]), 1)

    def test_calculate_determinant_invalid(self):
        with self.assertRaises(ValueError):
            calculate_determinant([[1, 2], [3]])

    # Tests for Fibonacci Sequence
    def test_calculate_fibonacci_sequence_count(self):
        self.assertEqual(calculate_fibonacci_sequence(count=10), [0, 1, 1, 2, 3, 5, 8, 13, 21, 34])

    def test_calculate_fibonacci_sequence_max_value(self):
        self.assertEqual(calculate_fibonacci_sequence(max_value=20), [0, 1, 1, 2, 3, 5, 8, 13])

    def test_calculate_fibonacci_sequence_invalid(self):
        with self.assertRaises(ValueError):
            calculate_fibonacci_sequence()

    # Tests for Matrix Operations
    def test_sum_matrices(self):
        self.assertEqual(sum_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[6, 8], [10, 12]])

    def test_product_matrices(self):
        self.assertEqual(product_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[19, 22], [43, 50]])

    def test_sum_matrices_invalid(self):
        with self.assertRaises(ValueError):
            sum_matrices([[1, 2]], [[3, 4], [5, 6]])

    # Tests for Pascal's Triangle
    def test_generate_pascals_triangle(self):
        expected = [
            [1],
            [1, 1],
            [1, 2, 1],
            [1, 3, 3, 1],
            [1, 4, 6, 4, 1]
        ]
        self.assertEqual(generate_pascals_triangle(5), expected)

    def test_generate_pascals_triangle_invalid(self):
        with self.assertRaises(ValueError):
            generate_pascals_triangle(0)

    # Tests for Vector Operations
    def test_sum_vectors(self):
        self.assertEqual(sum_vectors([1, 2, 3], [4, 5, 6]), [5, 7, 9])

    def test_sum_vectors_invalid(self):
        with self.assertRaises(ValueError):
            sum_vectors([1, 2], [1, 2, 3])

if __name__ == "__main__":
    unittest.main()