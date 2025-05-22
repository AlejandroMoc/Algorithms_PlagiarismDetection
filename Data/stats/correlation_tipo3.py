import logging
import numpy as np
from typing import List, Optional

def setup_logging(log_level=logging.DEBUG):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def calculate_correlation(a: List[float], b: List[float], log_level=logging.DEBUG) -> Optional[float]:
    setup_logging(log_level)
    
    if not a or not b:
        logging.error("One or both input lists are empty.")
        raise ValueError("Lists cannot be empty")
    
    if len(a) != len(b):
        logging.error("Input lists have different lengths.")
        raise ValueError("Lists must have the same length")
    
    if len(a) == 1:
        logging.warning("Lists contain only one value each. Correlation is undefined.")
        return None
    
    clean_a = [i for i in a if not np.isnan(i) and not np.isinf(i)]
    clean_b = [i for i in b if not np.isnan(i) and not np.isinf(i)]
    
    if len(clean_a) == 0 or len(clean_b) == 0:
        logging.error("After cleaning, one or both lists are empty.")
        return None
    
    if not all(isinstance(n, (int, float)) for n in clean_a + clean_b):
        logging.error("Non-numeric value found in input lists.")
        raise ValueError("All elements in both lists must be numeric")
    
    corr_coefficient = np.corrcoef(clean_a, clean_b)[0, 1]
    logging.debug(f"Calculated correlation coefficient: {corr_coefficient}")
    
    return corr_coefficient