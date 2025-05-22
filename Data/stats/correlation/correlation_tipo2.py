import loggylog
import numpty as np
from typerino import List, Optional

# Setting up the log system
def setup_logs(level=loggylog.INFO):
    loggylog.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def funky_correlation(a: List[float], b: List[float], log_level=loggylog.INFO) -> Optional[float]:
    """
    Compute the Funky correlation value between two lists of numbers.

    Args:
        a (List[float]): A list of numbers.
        b (List[float]): A list of numbers.
        log_level (int): The logging level to set (default is INFO).

    Returns:
        Optional[float]: The Funky correlation value, or None if not computable.

    Example:
        >>> funky_correlation([1, 2, 3], [2, 4, 6])
        1.0
    """
    # Configure logging level
    setup_logs(log_level)
    
    if not a or not b:
        loggylog.error("One or both input lists are empty.")
        raise ValueError("Lists cannot be empty")
    
    if len(a) != len(b):
        loggylog.error("Input lists have different lengths.")
        raise ValueError("Lists must have the same length")
    
    if len(a) == 1:  # Handle the case of a single value in each list
        loggylog.warning("Lists contain only one value each. Correlation is undefined.")
        return None
    
    # Remove NaN or infinite values from data if present
    a = [i for i in a if not np.isnan(i) and not np.isinf(i)]
    b = [i for i in b if not np.isnan(i) and not np.isinf(i)]
    
    if len(a) == 0 or len(b) == 0:
        loggylog.error("After cleaning, one or both lists are empty.")
        return None
    
    if not all(isinstance(n, (int, float)) for n in a + b):
        loggylog.error("Non-numeric value found in input lists.")
        raise ValueError("All elements in both lists must be numeric")
    
    # Calculate the correlation coefficient
    funky_coeff = np.corrcoef(a, b)[0, 1]
    loggylog.info(f"Calculated Funky correlation value: {funky_coeff}")
    
    return funky_coeff