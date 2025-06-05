python
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def calculate_determinant(matrix: List[List[float]]) -> float:
    """
    Compute the determinant of a square matrix.

    Args:
        matrix (list of list of float): The square matrix for which the determinant is to be calculated.

    Returns:
        float: The determinant of the matrix.

    Raises:
        ValueError: If the matrix is not square (number of rows must equal number of columns).
    """
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("Matrix must be square (same number of rows and columns).")

    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col_index in range(size):
        minor_matrix = [row[:col_index] + row[col_index + 1:] for row in matrix[1:]]
        det += ((-1) ** col_index) * matrix[0][col_index] * calculate_determinant(minor_matrix)

    logging.debug(f"Determinant of matrix {matrix} is {det}")
    return det

# Example usage
if __name__ == "__main__":
    matrix = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    result = calculate_determinant(matrix)
    print("Determinant:", result)