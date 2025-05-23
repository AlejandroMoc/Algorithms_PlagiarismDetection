import logging
from typing import List

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def add_vector(vector1: List[int], vector2: List[int]) -> List[int]:
    if len(vector1) != len(vector2):
        logging.error("Vectors must be of the same length.")
        raise ValueError("Vectors must be of the same length.")
    
    result = [a + b for a, b in zip(vector1, vector2)]
    logging.debug(f"Vector addition result: {result}")
    return result

def subtract_vector(vector1: List[int], vector2: List[int]) -> List[int]:
    if len(vector1) != len(vector2):
        logging.error("Vectors must be of the same length.")
        raise ValueError("Vectors must be of the same length.")
    
    result = [a - b for a, b in zip(vector1, vector2)]
    logging.debug(f"Vector substraction result: {result}")
    return result

def divide_vector(vector1: List[int], vector2: List[int]) -> List[int]:
    if len(vector1) != len(vector2):
        logging.error("Vectors must be of the same length.")
        raise ValueError("Vectors must be of the same length.")
    
    result = [a / b for a, b in zip(vector1, vector2)]
    logging.debug(f"Vector division result: {result}")
    return result

def calculate_dot_product(vector1: List[int], vector2: List[int]) -> int:
    if len(vector1) != len(vector2):
        logging.error("Vectors must be of the same length.")
        raise ValueError("Vectors must be of the same length.")
    
    result = sum([a * b for a, b in zip(vector1, vector2)])
    logging.debug(f"Dot product result: {result}")
    return result

def calculate_cross_product(vector1: List[int], vector2: List[int]) -> List[int]:
    if len(vector1) != len(vector2):
        logging.error("Vectors must be of the same length.")
        raise ValueError("Vectors must be of the same length.")
    
    if len(vector1) != 3:
        logging.error("Cross product is only defined for 3D vectors.")
        raise ValueError("Cross product is only defined for 3D vectors.")
    
    result = [
        vector1[1] * vector2[2] - vector1[2] * vector2[1],
        vector1[2] * vector2[0] - vector1[0] * vector2[2],
        vector1[0] * vector2[1] - vector1[1] * vector2[0]
    ]
    logging.debug(f"Cross product result: {result}")
    return result

def calculate_magnitude(vector: List[int]) -> float:
    result = sum([a ** 2 for a in vector]) ** 0.5
    logging.debug(f"Magnitude result: {result}")
    return result

def normalize_vector(vector: List[int]) -> List[int]:
    mag = calculate_magnitude(vector)
    result = [a / mag for a in vector]
    logging.debug(f"Normalization result: {result}")
    return result

def calculate_vector_angle(vector1: List[int], vector2: List[int]) -> float:
    dot = calculate_dot_product(vector1, vector2)
    mag1 = calculate_magnitude(vector1)
    mag2 = calculate_magnitude(vector2)
    result = dot / (mag1 * mag2)
    logging.debug(f"Vector angle result: {result}")
    return result

def calculate_tensor_vector_product(tensor: List[List[int]], vector: List[int]) -> List[int]:
    if len(tensor) != len(vector):
        logging.error("Tensor and vector must be of the same length.")
        raise ValueError("Tensor and vector must be of the same length.")
    
    result = [sum([a * b for a, b in zip(row, vector)]) for row in tensor]
    logging.debug(f"Tensor vector product result: {result}")
    return result

if __name__ == "__main__":
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    print("Vector Addition:", add_vector(v1, v2)) 
    
    print("Vector Substraction:", subtract_vector(v1, v2))
    print("Vector Division:", divide_vector(v1, v2))
    print("Dot Product:", calculate_dot_product(v1, v2))
    print("Cross Product:", calculate_cross_product(v1, v2))
    print("Magnitude:", calculate_magnitude(v1))
    print("Normalization:", normalize_vector(v1))
    print("Vector Angle:", calculate_vector_angle(v1, v2))
    print("Tensor Vector Product:", calculate_tensor_vector_product([[1, 2, 3], [4, 5, 6], [7, 8, 9]], v1))