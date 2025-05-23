import logging
from typing import List
from mean_median_mode import mean as m
from standard_deviation import standard_deviation as sd

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def zeta_transform(numbers_list: List[float]) -> List[float]:
    """
    Normalize a list of numbers using Z-score normalization.

    Args:
        numbers_list (List[float]): A list of numbers.

    Returns:
        List[float]: The normalized values.

    Example:
        >>> zeta_transform([1, 2, 3, 4, 5])
        [-1.4142135623730951, -0.7071067811865475, 0.0, 0.7071067811865475, 1.4142135623730951]
    """
    if not numbers_list:
        logging.error("Input data list is empty.")
        raise ValueError("List is empty")
    
    if not all(isinstance(x, (int, float)) for x in numbers_list):
        logging.error("Non-numeric value found in data list.")
        raise ValueError("All elements in the data list must be numeric")
    
    mean_value = m(numbers_list)
    std_dev = sd(numbers_list)

    # Handle the case where standard deviation is zero
    if std_dev == 0:
        logging.warning("Standard deviation is zero; all values are identical.")
        return [0] * len(numbers_list)  # Return zeros since all values are the same
    
    transformed_data = [(x - mean_value) / std_dev for x in numbers_list]
    logging.debug(f"Calculated Z-score normalized data: {transformed_data}")
    return transformed_data