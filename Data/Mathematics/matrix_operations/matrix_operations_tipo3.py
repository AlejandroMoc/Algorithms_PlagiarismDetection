import logging
from typing import List
import numpy as np

logging.basicConfig(level=logging.DEBUG)

def add_matrices(matrix1: List[List[float]], matrix2: List[List[float]]) -> List[List[float]]:
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("The dimensions of the two matrices must be the same.")

    result = [[matrix1[i][j] + matrix2[i][j] for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
    logging.debug(f"Added matrices {matrix1} and {matrix2} to get {result}")
    return result

def multiply_matrices(matrix1: List[List[float]], matrix2: List[List[float]]) -> List[List[float]]:
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Number of columns in first matrix must match number of rows in second matrix.")

    result = [[sum(matrix1[i][k] * matrix2[k][j] for k in range(len(matrix2))) for j in range(len(matrix2[0]))] for i in range(len(matrix1))]
    logging.debug(f"Multiplied matrices {matrix1} and {matrix2} to get {result}")
    return result

def transpose_matrix(matrix):
    result = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    logging.debug(f"Transposed matrix {matrix} to get {result}")
    return result

def scalar_multiply_matrix(matrix, scalar):
    result = [[matrix[i][j] * scalar for j in range(len(matrix[0]))] for i in range(len(matrix))]
    logging.debug(f"Multiplied matrix {matrix} by scalar {scalar} to get {result}")
    return result

def calculate_determinant(matrix):
    if len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be square to compute determinant.")

    if len(matrix) == 1:
        return matrix[0][0]
    elif len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        det = 0
        for i in range(len(matrix)):
            minor = [row[:i] + row[i+1:] for row in matrix[1:]]
            det += ((-1) ** i) * matrix[0][i] * calculate_determinant(minor)
        return det

def calculate_inverse(matrix):
    if len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be square to compute inverse.")

    det = calculate_determinant(matrix)
    if det == 0:
        raise ValueError("Matrix is singular and does not have an inverse.")

    adjugate = [[((-1) ** (i + j)) * calculate_determinant([row[:j] + row[j+1:] for row in matrix[:i] + matrix[i+1:]]) for j in range(len(matrix))] for i in range(len(matrix))]
    inverse = scalar_multiply_matrix(transpose_matrix(adjugate), 1 / det)
    
    return inverse

def calculate_trace(matrix):
    if len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be square to compute trace.")

    return sum(matrix[i][i] for i in range(len(matrix)))

def calculate_eigenvalues_and_eigenvectors(matrix):
    if len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be square to compute eigenvalues and eigenvectors.")

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

if __name__ == "__main__":
    matrix_A = [[1, 2], [3, 4]]
    matrix_B = [[5, 6], [7, 8]]

    print("Matrix Addition:")
    print(add_matrices(matrix_A, matrix_B))

    print("\nMatrix Multiplication:")
    print(multiply_matrices(matrix_A, matrix_B))

    print("\nMatrix Transpose:")
    print(transpose_matrix(matrix_A))

    print("\nMatrix Scalar Multiplication:")
    print(scalar_multiply_matrix(matrix_A, 2))

    print("\nDeterminant:")
    print(calculate_determinant(matrix_A))

    print("\nMatrix Inverse:")
    print(calculate_inverse(matrix_A))

    print("\nMatrix Trace:")
    print(calculate_trace(matrix_A))

    print("\nEigenvalues and Eigenvectors:")
    eigenvalues, eigenvectors = calculate_eigenvalues_and_eigenvectors(matrix_A)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:", eigenvectors)