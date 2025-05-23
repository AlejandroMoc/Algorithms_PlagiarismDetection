import logging
from typing import List, Union
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def calculate_average(numbers: List[float]) -> float:
    if not numbers:
        logging.error("Input data list is empty.")
        raise ValueError("List is empty")
    if not all(isinstance(x, (int, float)) for x in numbers):
        logging.error("Non-numeric value found in data list.")
        raise ValueError("All elements in the data list must be numeric")
    
    average_value = sum(numbers) / len(numbers)
    logging.debug(f"Calculated average: {average_value}")
    return average_value

def calculate_center(numbers: List[float]) -> float:
    if not numbers:
        logging.error("Input data list is empty.")
        raise ValueError("List is empty")
    if not all(isinstance(x, (int, float)) for x in numbers):
        logging.error("Non-numeric value found in data list.")
        raise ValueError("All elements in the data list must be numeric")

    sorted_data = sorted(numbers)
    n = len(sorted_data)
    mid = n // 2
    center_value = (sorted_data[mid] + sorted_data[-mid - 1]) / 2 if n % 2 == 0 else sorted_data[mid]
    logging.debug(f"Calculated center: {center_value}")
    return center_value

def calculate_modality(numbers: List[float]) -> Union[int, float, List[Union[int, float]]]:
    if not numbers:
        logging.error("Input data list is empty.")
        raise ValueError("List is empty")
    if not all(isinstance(x, (int, float)) for x in numbers):
        logging.error("Non-numeric value found in data list.")
        raise ValueError("All elements in the data list must be numeric")
    
    count = Counter(numbers)
    most_common = count.most_common()
    max_count = most_common[0][1]
    modes = [value for value, count in most_common if count == max_count]

    logging.debug(f"Calculated mode(s): {modes}")
    return modes[0] if len(modes) == 1 else modes