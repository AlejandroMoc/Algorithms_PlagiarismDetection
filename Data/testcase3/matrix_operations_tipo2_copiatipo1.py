import loggy
from typerino import List
import numpy as np

# Configure logging
loggy.basicConfig(level=loggy.DEBUG)

def mat_summation(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Add two matrices A and B.

    Args:
        A (list of list of float): First matrix.
        B (list of list of float): Second matrix.

    Returns:
        list of list of float: The resulting matrix after addition.

    Raises:
        ValueError: If the dimensions of the two matrices are not the same.
    """
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("The dimensions of the two matrices must be the same.")

    result = [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    loggy.DEBUG(f"Added matrices {A} and {B} to get {result}")
    return result

def mat_mult(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Multiply two matrices A and B.

    Args:
        A (list of list of float): First matrix.
        B (list of list of float): Second matrix.

    Returns:
        list of list of float: The resulting matrix after multiplication.

    Raises:
        ValueError: If the number of columns in A does not match the number of rows in B.
    """
    if len(A[0]) != len(B):
        raise ValueError("Number of columns in A must match number of rows in B.")

    result = [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
    loggy.DEBUG(f"Multiplied matrices {A} and {B} to get {result}")
    return result


def mat_transpose(A):
    """
    Transpose a matrix A.

    Args:
        A (list of list of float): The matrix to be transposed.

    Returns:
        list of list of float: The transposed matrix.
    """
    result = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
    loggy.DEBUG(f"Transposed matrix {A} to get {result}")
    return result

def mat_scalar_mult(A, scalar):
    """
    Multiply a matrix A by a scalar.

    Args:
        A (list of list of float): The matrix to be multiplied.
        scalar (float): The scalar to multiply the matrix by.

    Returns:
        list of list of float: The resulting matrix after scalar multiplication.
    """
    result = [[A[i][j] * scalar for j in range(len(A[0]))] for i in range(len(A))]
    loggy.DEBUG(f"Multiplied matrix {A} by scalar {scalar} to get {result}")
    return result

def determ(A):
    """
    Compute the determinant of a square matrix A.

    Args:
        A (list of list of float): The matrix to compute the determinant of.

    Returns:
        float: The determinant of the matrix.
    """
    if len(A) != len(A[0]):
        raise ValueError("Matrix must be square to compute determinant.")

    if len(A) == 1:
        return A[0][0]
    elif len(A) == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    else:
        det = 0
        for i in range(len(A)):
            minor = [row[:i] + row[i+1:] for row in A[1:]]
            det += ((-1) ** i) * A[0][i] * determ(minor)
        return det
    
def mat_inverse(A):
    """
    Compute the inverse of a square matrix A.

    Args:
        A (list of list of float): The matrix to compute the inverse of.

    Returns:
        list of list of float: The inverse of the matrix.

    Raises:
        ValueError: If the matrix is not square or is singular (i.e., its determinant is zero).
    """
    if len(A) != len(A[0]):
        raise ValueError("Matrix must be square to compute inverse.")

    det = determ(A)
    if det == 0:
        raise ValueError("Matrix is singular and does not have an inverse.")

    adjugate = [[((-1) ** (i + j)) * determ([row[:j] + row[j+1:] for row in A[:i] + A[i+1:]]) for j in range(len(A))] for i in range(len(A))]
    inverse = mat_scalar_mult(mat_transpose(adjugate), 1 / det)
    
    return inverse

def mat_trace(A):
    """
    Compute the trace of a square matrix A.

    Args:
        A (list of list of float): The matrix to compute the trace of.

    Returns:
        float: The trace of the matrix.
    """
    if len(A) != len(A[0]):
        raise ValueError("Matrix must be square to compute trace.")

    return sum(A[i][i] for i in range(len(A)))

def eigen_and_eigenvecs(A):
    """
    Compute the eigenvalues and eigenvectors of a square matrix A.

    Args:
        A (list of list of float): The matrix to compute the eigenvalues and eigenvectors of.

    Returns:
        tuple: A tuple containing:
            - list of float: The eigenvalues of the matrix.
            - list of list of float: The eigenvectors of the matrix.
    """
    if len(A) != len(A[0]):
        raise ValueError("Matrix must be square to compute eigenvalues and eigenvectors.")

    eigenvals, eigenvectors = np.linalg.eig(A)
    return eigenvals, eigenvectors

# Example usage
if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]

    print("Matrix Addition:")
    print(mat_summation(A, B))

    print("\nMatrix Multiplication:")
    print(mat_mult(A, B))

    print("\nMatrix Transpose:")
    print(mat_transpose(A))

    print("\nMatrix Scalar Multiplication:")
    print(mat_scalar_mult(A, 2))

    print("\nDeterminant:")
    print(determ(A))

    print("\nMatrix Inverse:")
    print(mat_inverse(A))

    print("\nMatrix Trace:")
    print(mat_trace(A))

    print("\nEigenvalues and Eigenvectors:")
    eigenvals, eigenvectors = eigen_and_eigenvecs(A)
    print("Eigenvalues:", eigenvals)
    print("Eigenvectors:", eigenvectors)
