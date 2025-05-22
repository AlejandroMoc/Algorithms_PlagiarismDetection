import logging
from typing import List
from mean_median_mode import mean
from standard_deviation import standard_deviation

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def z_score_normalization(data_list: List[float]) -> List[float]:
    if not data_list:
        logging.error("Input data list is empty.")
        raise ValueError("List is empty")
    
    if not all(isinstance(value, (int, float)) for value in data_list):
        logging.error("Non-numeric value found in data list.")
        raise ValueError("All elements in the data list must be numeric")
    
    mean_value = mean(data_list)
    std_dev = standard_deviation(data_list)

    if std_dev == 0:
        logging.warning("Standard deviation is zero; all values are identical.")
        return [0] * len(data_list)
    
    normalized_data = [(value - mean_value) / std_dev for value in data_list]
    logging.debug(f"Calculated Z-score normalized data: {normalized_data}")
    return normalized_data