python
import logging
import numpy as np
from typing import List

def setup_logging(log_level=logging.DEBUG):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def calculate_kurtosis(data: List[float], log_level=logging.DEBUG) -> float:
    setup_logging(log_level)

    if not data:
        logging.error("Input data list is empty.")
        raise ValueError("List is empty")
    
    if not all(isinstance(x, (int, float)) for x in data):
        logging.error("Non-numeric value found in data list.")
        raise ValueError("All elements in the data list must be numeric")

    if len(data) < 4:
        logging.warning("Data has fewer than 4 elements, kurtosis calculation may not be meaningful.")
        return None

    cleaned_data = [x for x in data if not np.isnan(x) and not np.isinf(x)]
    if len(cleaned_data) < 4:
        logging.warning("After cleaning, data has fewer than 4 valid elements.")
        return None

    mean_val = np.mean(cleaned_data)
    std_dev = np.std(cleaned_data)
    
    if std_dev == 0:
        logging.warning("Standard deviation is zero, kurtosis is undefined.")
        return None
    
    kurtosis_val = np.mean((cleaned_data - mean_val) ** 4) / (std_dev ** 4) - 3
    logging.debug(f"Calculated kurtosis: {kurtosis_val}")
    return kurtosis_val