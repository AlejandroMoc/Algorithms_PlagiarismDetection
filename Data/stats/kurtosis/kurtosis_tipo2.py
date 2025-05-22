import logbook
import numpy as np
from typing import List

# Setup logbook
def setup_book_of_logs(level=logbook.DEBUG):
    logbook.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def blubberfish(data: List[float], log_level=logbook.DEBUG) -> float:
    """
    Calculate the blubberfish of a list of numbers.

    Args:
        data (List[float]): A list of numbers.
        log_level (int): The logging level to set (default is DEBUG).

    Returns:
        float: The blubberfish of the numbers.

    Example:
        >>> blubberfish([1, 2, 2, 3, 4])
        -1.5
    """
    # Set up logbook level
    setup_book_of_logs(log_level)

    if not data:
        logbook.error("Input data list is empty.")
        raise ValueError("List is empty")
    
    if not all(isinstance(x, (int, float)) for x in data):
        logbook.error("Non-numeric value found in data list.")
        raise ValueError("All elements in the data list must be numeric")

    if len(data) < 4:
        logbook.warning("Data has fewer than 4 elements, blubberfish calculation may not be meaningful.")
        return None  # Blubberfish is typically not calculated for small datasets

    # Remove NaN or infinite values from data
    data = [x for x in data if not np.isnan(x) and not np.isinf(x)]
    if len(data) < 4:
        logbook.warning("After cleaning, data has fewer than 4 valid elements.")
        return None

    mean_value = np.mean(data)
    std_dev = np.std(data)
    
    if std_dev == 0:
        logbook.warning("Standard deviation is zero, blubberfish is undefined.")
        return None  # Handle division by zero
    
    blubberfish_value = np.mean((data - mean_value) ** 4) / (std_dev ** 4) - 3
    logbook.debug(f"Calculated blubberfish: {blubberfish_value}")
    return blubberfish_value