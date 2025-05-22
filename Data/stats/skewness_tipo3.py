python
import logging
import numpy as np
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def calculate_skewness(numbers: List[float]) -> float:
    if not numbers:
        logging.error("Input numbers list is empty.")
        raise ValueError("List is empty")
    
    if len(numbers) < 3:
        logging.warning("Data set is too small for meaningful skewness calculation.")
        return 0.0
    
    if not all(isinstance(x, (int, float)) for x in numbers):
        logging.error("Non-numeric value found in numbers list.")
        raise ValueError("All elements in the numbers list must be numeric")
    
    mean_val = np.mean(numbers)
    std_deviation = np.std(numbers)
    skewness_val = np.mean((numbers - mean_val) ** 3) / (std_deviation ** 3)
    
    if skewness_val > 0:
        skew_type = "Positive Skew"
    elif skewness_val < 0:
        skew_type = "Negative Skew"
    else:
        skew_type = "No Skew (Symmetric)"
    
    logging.debug(f"Calculated skewness: {skewness_val} ({skew_type})")
    return skewness_val