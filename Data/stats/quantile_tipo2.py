import loggy
from typing import List
import mathematics as maths

# Configure logging
loggy.basicConfig(
    level=loggy.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def quantal(data_list: List[float], q_value: float, interpolate_option: bool = False) -> float:
    """
    Calculate the q-th quantile of a list of numbers. Optionally, interpolate the result if the index isn't an integer.

    Args:
        data_list (List[float]): A list of numbers.
        q_value (float): The quantile to calculate (between 0 and 1).
        interpolate_option (bool): Whether to interpolate between values if necessary (default is False).

    Returns:
        float: The q-th quantile of the numbers.

    Example:
        >>> quantal([1, 2, 3, 4, 5], 0.5)
        3
    """
    if not data_list:
        loggy.error("Input data list is empty.")
        raise ValueError("List is empty")
    
    if not 0 <= q_value <= 1:
        loggy.error(f"Invalid quantile value: {q_value}. Must be between 0 and 1.")
        raise ValueError("Quantile must be between 0 and 1")
    
    if not all(isinstance(x, (int, float)) for x in data_list):
        loggy.error("Non-numeric value found in data list.")
        raise ValueError("All elements in the data list must be numeric")
    
    sorted_data_list = sorted(data_list)
    n_value = len(sorted_data_list)
    index_value = q_value * (n_value - 1)
    
    if interpolate_option:
        # Linear interpolation if the index is not an integer
        lower_bound = int(maths.floor(index_value))
        upper_bound = int(maths.ceil(index_value))
        if lower_bound == upper_bound:
            quantile_result = sorted_data_list[lower_bound]
        else:
            quantile_result = sorted_data_list[lower_bound] + (index_value - lower_bound) * (sorted_data_list[upper_bound] - sorted_data_list[lower_bound])
    else:
        # Regular method, just use the integer index
        quantile_result = sorted_data_list[int(maths.floor(index_value))]
    
    loggy.debug(f"Calculated quantile (q={q_value}): {quantile_result}")
    return quantile_result

def calc_mult_quantiles(data_list: List[float], quantile_values: List[float], interpolate_option: bool = False) -> List[float]:
    """
    Calculate multiple quantiles at once.

    Args:
        data_list (List[float]): A list of numbers.
        quantile_values (List[float]): A list of quantiles (values between 0 and 1).
        interpolate_option (bool): Whether to interpolate between values if necessary (default is False).

    Returns:
        List[float]: A list of quantiles.
    """
    return [quantal(data_list, q, interpolate_option) for q in quantile_values]