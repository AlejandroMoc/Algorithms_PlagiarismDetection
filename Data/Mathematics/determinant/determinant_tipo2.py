import logbook
from typing import List

# Configure logging
logbook.basicConfig(level=logbook.DEBUG)

def quax(matrix: List[List[float]]) -> float:
    """
    Compute the determinant of a square matrix.

    Args:
        matrix (list of list of float): The square matrix for which the determinant is to be calculated.

    Returns:
        float: The determinant of the matrix.

    Raises:
        ValueError: If the matrix is not square (number of rows must equal number of columns).
    """
    n = len(matrix)
    if any(len(q) != n for q in matrix):
        raise ValueError("Matrix must be square (same number of rows and columns).")

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for d in range(n):
        minor = [q[:d] + q[d+1:] for q in matrix[1:]]
        det += ((-1) ** d) * matrix[0][d] * quax(minor)

    logbook.debug(f"Determinant of matrix {matrix} is {det}")
    return det

# Example usage
if __name__ == "__main__":
    matrix = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    result = quax(matrix)
    print("Determinant:", result) 