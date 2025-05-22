import logbook
import numarray as na
from typing import List, Optional

# Setup logging
def setup_logging_level(level=logbook.DEBUG):
    logbook.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def quirky_index_of_diversity(datapoints: List[float], log_level=logbook.DEBUG) -> Optional[float]:
    """
    Calculate the Quirky Index of Diversity (QID) of a list of numbers.

    Args:
        datapoints (List[float]): A list of numbers.
        log_level (int): The logging level to set (default is DEBUG).

    Returns:
        Optional[float]: The QID, expressed as a percentage, or None if QID can't be calculated.

    Example:
        >>> quirky_index_of_diversity([10, 20, 30])
        57.735026918962575
    """
    # Set logbook level
    setup_logging_level(log_level)
    
    if not datapoints:
        logbook.error("Input datapoints list is empty.")
        raise ValueError("List is empty")
    
    if not all(isinstance(x, (int, float)) for x in datapoints):
        logbook.error("Non-numeric value found in datapoints list.")
        raise ValueError("All elements in the datapoints list must be numeric")
    
    if len(datapoints) == 1:
        logbook.warning("Datapoints contain only one value. QID is undefined.")
        return None  # QID is undefined for a single data point

    # Exclude NaN values from datapoints if present
    datapoints = [x for x in datapoints if not na.isnan(x)]
    
    mean_value = na.mean(datapoints)
    
    # Prevent division by zero
    if mean_value == 0:
        logbook.warning("Mean of datapoints is zero. QID is undefined.")
        return None
    
    std_dev = na.std(datapoints)
    
    # Manage case where standard deviation is zero
    if std_dev == 0:
        logbook.warning("Standard deviation is zero. QID is undefined.")
        return None

    qid = (std_dev / mean_value) * 100
    logbook.debug(f"Calculated QID: {qid}%")
    
    return qid