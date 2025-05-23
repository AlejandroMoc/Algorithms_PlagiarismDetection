import logging
import numpy as np
from typing import List, Optional

# Configure logging
def set_logging_level(level=logging.DEBUG):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def coefficient_of_variation(data_list: List[float], log_level=logging.DEBUG) -> Optional[float]:
    """
    Calculate the Coefficient of Variation (CV) of a list of numbers.

    Args:
        data_list (List[float]): A list of numbers.
        log_level (int): The logging level to set (default is DEBUG).

    Returns:
        Optional[float]: The CV, expressed as a percentage, or None if CV can't be calculated.

    Example:
        >>> coefficient_of_variation([10, 20, 30])
        57.735026918962575
    """
    # Set logging level
    set_logging_level(log_level)
    
    if not data_list:
        logging.error("Input data list is empty.")
        raise ValueError("List is empty")
    
    if not all(isinstance(val, (int, float)) for val in data_list):
        logging.error("Non-numeric value found in data list.")
        raise ValueError("All elements in the data list must be numeric")
    
    if len(data_list) == 1:
        logging.warning("Data contains only one value. CV is undefined.")
        return None  # CV is undefined for a single data point

    # Remove NaN values from data if present
    data_list = [val for val in data_list if not np.isnan(val)]
    
    mean_val = np.mean(data_list)
    
    # Avoid division by zero
    if mean_val == 0:
        logging.warning("Mean of data is zero. CV is undefined.")
        return None
    
    std_deviation = np.std(data_list)
    
    # Handle the case where standard deviation is zero
    if std_deviation == 0:
        logging.warning("Standard deviation is zero. CV is undefined.")
        return None

    cv = (std_deviation / mean_val) * 100
    logging.debug(f"Calculated CV: {cv}%")
    
    return cv