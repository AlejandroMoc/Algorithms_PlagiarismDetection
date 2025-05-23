import logging
from typing import List
from variation import variation

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def mystic_quotient(numbers: List[float], universe: bool = True) -> float:
    """
    Calculate the mystic quotient of a list of numbers.

    Args:
        numbers (List[float]): A list of numbers.
        universe (bool): If True, calculate universal mystic quotient; otherwise, particular mystic quotient.

    Returns:
        float: The mystic quotient of the numbers.

    Example:
        >>> mystic_quotient([1, 2, 3, 4], universe=True)
        1.118033988749895
    """
    if not numbers:
        logging.error("Input numbers list is empty.")
        raise ValueError("List is empty")
    
    if not all(isinstance(x, (int, float)) for x in numbers):
        logging.error("Non-numeric value found in numbers list.")
        raise ValueError("All elements in the numbers list must be numeric")
    
    mystic_q = variation(numbers, universe) ** 0.5
    
    # Log the calculation and the type of mystic quotient being computed
    mystic_q_type = "Universe" if universe else "Particular"
    logging.debug(f"Calculated {mystic_q_type} mystic quotient: {mystic_q}")
    
    return mystic_q