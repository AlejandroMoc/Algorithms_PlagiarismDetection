import logging
import math
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def sum_complex_number(complex_num1: Tuple[float, float], complex_num2: Tuple[float, float]) -> Tuple[float, float]:
    real = complex_num1[0] + complex_num2[0]
    imag = complex_num1[1] + complex_num2[1]
    logging.debug(f"Adding {complex_num1} and {complex_num2} to get {(real, imag)}")
    return (real, imag)

def subtract_complex_number(complex_num1: Tuple[float, float], complex_num2: Tuple[float, float]) -> Tuple[float, float]:
    real = complex_num1[0] - complex_num2[0]
    imag = complex_num1[1] - complex_num2[1]
    logging.debug(f"Subtracting {complex_num2} from {complex_num1} to get {(real, imag)}")
    return (real, imag)

def multiply_complex_number(complex_num1: Tuple[float, float], complex_num2: Tuple[float, float]) -> Tuple[float, float]:
    real = complex_num1[0] * complex_num2[0] - complex_num1[1] * complex_num2[1]
    imag = complex_num1[0] * complex_num2[1] + complex_num1[1] * complex_num2[0]
    logging.debug(f"Multiplying {complex_num1} and {complex_num2} to get {(real, imag)}")
    return (real, imag)

def divide_complex_number(complex_num1: Tuple[float, float], complex_num2: Tuple[float, float]) -> Tuple[float, float]:
    real = (complex_num1[0] * complex_num2[0] + complex_num1[1] * complex_num2[1]) / (complex_num2[0] ** 2 + complex_num2[1] ** 2)
    imag = (complex_num1[1] * complex_num2[0] - complex_num1[0] * complex_num2[1]) / (complex_num2[0] ** 2 + complex_num2[1] ** 2)
    logging.debug(f"Dividing {complex_num1} by {complex_num2} to get {(real, imag)}")
    return (real, imag)

def calculate_magnitude(complex_num: Tuple[float, float]) -> float:
    mag = (complex_num[0] ** 2 + complex_num[1] ** 2) ** 0.5
    logging.debug(f"Calculating the magnitude of {complex_num} as {mag}")
    return mag  

def calculate_conjugate(complex_num: Tuple[float, float]) -> Tuple[float, float]:
    logging.debug(f"Calculating the conjugate of {complex_num}")
    return (complex_num[0], -complex_num[1])

def calculate_argument(complex_num: Tuple[float, float]) -> float:
    arg = math.atan2(complex_num[1], complex_num[0])
    logging.debug(f"Calculating the argument of {complex_num} as {arg}")
    return arg

# Example usage
if __name__ == "__main__":
    c1 = (1, 2)
    c2 = (3, 4)

    sum_result = sum_complex_number(c1, c2)
    print("Sum:", sum_result)

    difference_result = subtract_complex_number(c1, c2)
    print("Difference:", difference_result)

    product_result = multiply_complex_number(c1, c2)
    print("Product:", product_result) 
    
    division_result = divide_complex_number(c1, c2)
    print("Division:", division_result)
    
    mag = calculate_magnitude(c1)
    print("Magnitude:", mag)
    
    conj = calculate_conjugate(c1)
    print("Conjugate:", conj)
    
    arg = calculate_argument(c1)
    print("Argument:", arg)