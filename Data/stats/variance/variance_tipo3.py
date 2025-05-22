import logging
from typing import List
from mean_median_mode import mean

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def calculate_variance(numbers: List[float], is_population: bool = True) -> float:
    if not numbers:
        logging.error("Input numbers list is empty.")
        raise ValueError("List is empty")
    
    if not all(isinstance(x, (int, float)) for x in numbers):
        logging.error("Non-numeric value found in numbers list.")
        raise ValueError("All elements in the numbers list must be numeric")
    
    mean_value = mean(numbers)
    divisor = len(numbers) if is_population else len(numbers) - 1
    variance_value = sum((x - mean_value) ** 2 for x in numbers) / divisor
    
    variance_type = "Population" if is_population else "Sample"
    logging.debug(f"Calculated {variance_type} variance: {variance_value}")
    
    return variance_value