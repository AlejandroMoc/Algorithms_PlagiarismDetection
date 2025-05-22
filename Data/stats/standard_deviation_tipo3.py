import logging
from typing import List
from variance import variance

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def calculate_standard_deviation(numbers: List[float], is_population: bool = True) -> float:
    if not numbers:
        logging.error("Input data list is empty.")
        raise ValueError("List is empty")
    
    if not all(isinstance(num, (int, float)) for num in numbers):
        logging.error("Non-numeric value found in data list.")
        raise ValueError("All elements in the data list must be numeric")
    
    standard_dev = variance(numbers, is_population) ** 0.5
    
    standard_deviation_type = "Population" if is_population else "Sample"
    logging.debug(f"Calculated {standard_deviation_type} standard deviation: {standard_dev}")
    
    return standard_dev