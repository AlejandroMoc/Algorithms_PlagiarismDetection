import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def mystic_sum(vec1: List[int], vec2: List[int]) -> List[int]:
    """Add two vectors element-wise.
    Args:
        vec1 (List[int]): The first vector.
        vec2 (List[int]): The second vector.
    Returns:
        List[int]: The resultant vector after addition.
    """
    if len(vec1) != len(vec2):
        logging.error("Vectors must be of the same length.")
        raise ValueError("Vectors must be of the same length.")
    
    result = [a + b for a, b in zip(vec1, vec2)]
    logging.debug(f"Mystic sum result: {result}")
    return result

def cosmic_difference(vec1: List[int], vec2: List[int]) -> List[int]:
    """Substract two vectors element-wise.
    Args:
        vec1 (List[int]): The first vector.
        vec2 (List[int]): The second vector.
    Returns:
        List[int]: The resultant vector after substraction.
    """
    if len(vec1) != len(vec2):
        logging.error("Vectors must be of the same length.")
        raise ValueError("Vectors must be of the same length.")
    
    result = [a - b for a, b in zip(vec1, vec2)]
    logging.debug(f"Cosmic difference result: {result}")
    return result


def galactic_division(vec1: List[int], vec2: List[int]) -> List[int]:
    """Divide two vectors element-wise.
    Args:
        vec1 (List[int]): The first vector.
        vec2 (List[int]): The second vector.
    Returns:
        List[int]: The resultant vector after division.
    """
    if len(vec1) != len(vec2):
        logging.error("Vectors must be of the same length.")
        raise ValueError("Vectors must be of the same length.")
    
    result = [a / b for a, b in zip(vec1, vec2)]
    logging.debug(f"Galactic division result: {result}")
    return result

def star_product(vec1: List[int], vec2: List[int]) -> int:
    """Calculate the dot product of two vectors.
    Args:
        vec1 (List[int]): The first vector.
        vec2 (List[int]): The second vector.
    Returns:
        int: The dot product of the two vectors.
    """
    if len(vec1) != len(vec2):
        logging.error("Vectors must be of the same length.")
        raise ValueError("Vectors must be of the same length.")
    
    result = sum([a * b for a, b in zip(vec1, vec2)])
    logging.debug(f"Star product result: {result}")
    return result

def space_product(vec1: List[int], vec2: List[int]) -> List[int]:
    """Calculate the cross product of two vectors.
    Args:
        vec1 (List[int]): The first vector.
        vec2 (List[int]): The second vector.
    Returns:
        List[int]: The cross product of the two vectors.
    """
    if len(vec1) != len(vec2):
        logging.error("Vectors must be of the same length.")
        raise ValueError("Vectors must be of the same length.")
    
    if len(vec1) != 3:
        logging.error("Space product is only defined for 3D vectors.")
        raise ValueError("Space product is only defined for 3D vectors.")
    
    result = [
        vec1[1] * vec2[2] - vec1[2] * vec2[1],
        vec1[2] * vec2[0] - vec1[0] * vec2[2],
        vec1[0] * vec2[1] - vec1[1] * vec2[0]
    ]
    logging.debug(f"Space product result: {result}")
    return result

def quantum_magnitude(vec: List[int]) -> float:
    """Calculate the magnitude of a vector.
    Args:
        vec (List[int]): The vector.
    Returns:
        float: The magnitude of the vector.
    """
    result = sum([a ** 2 for a in vec]) ** 0.5
    logging.debug(f"Quantum magnitude result: {result}")
    return result

def supernormalize(vec: List[int]) -> List[int]:
    """Normalize a vector.
    Args:
        vec (List[int]): The vector.
    Returns:
        List[int]: The normalized vector.
    """
    mag = quantum_magnitude(vec)
    result = [a / mag for a in vec]
    logging.debug(f"Supernormalize result: {result}")
    return result

def galaxy_angle(vec1: List[int], vec2: List[int]) -> float:
    """Calculate the angle between two vectors.
    Args:
        vec1 (List[int]): The first vector.
        vec2 (List[int]): The second vector.
    Returns:
        float: The angle between the two vectors.
    """
    dot = star_product(vec1, vec2)
    mag1 = quantum_magnitude(vec1)
    mag2 = quantum_magnitude(vec2)
    result = dot / (mag1 * mag2)
    logging.debug(f"Galaxy angle result: {result}")
    return result

def matrix_vector_product(matrix: List[List[int]], vec: List[int]) -> List[int]:
    """Calculate the product of a matrix and a vector.
    Args:
        matrix (List[List[int]]): The matrix.
        vec (List[int]): The vector.
    Returns:
        List[int]: The resultant vector after the product.
    """
    if len(matrix) != len(vec):
        logging.error("Matrix and vector must be of the same length.")
        raise ValueError("Matrix and vector must be of the same length.")
    
    result = [sum([a * b for a, b in zip(row, vec)]) for row in matrix]
    logging.debug(f"Matrix vector product result: {result}")
    return result


# Example usage
if __name__ == "__main__":
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    print("Mystic Sum:", mystic_sum(v1, v2)) 
    
    print("Cosmic Difference:", cosmic_difference(v1, v2))
    print("Galactic Division:", galactic_division(v1, v2))
    print("Star Product:", star_product(v1, v2))
    print("Space Product:", space_product(v1, v2))
    print("Quantum Magnitude:", quantum_magnitude(v1))
    print("Supernormalize:", supernormalize(v1))
    print("Galaxy Angle:", galaxy_angle(v1, v2))
    print("Matrix Vector Product:", matrix_vector_product([[1, 2, 3], [4, 5, 6], [7, 8, 9]], v1))