python
import logging
from typing import List, Optional

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def interpolation_search_modified(array: List[int], target_element: int) -> int:
    logging.info("Starting interpolation search")
    if not array:
        logging.warning("Empty array provided.")
        return -1
    
    low_idx, high_idx = 0, len(array) - 1
    
    while low_idx <= high_idx and array[low_idx] <= target_element <= array[high_idx]:
        if array[high_idx] == array[low_idx]:
            if array[low_idx] == target_element:
                logging.info(f"Target {target_element} found at index {low_idx}")
                return low_idx
            logging.warning(f"Target {target_element} not found in the array.")
            return -1

        position = low_idx + ((target_element - array[low_idx]) * (high_idx - low_idx) // (array[high_idx] - array[low_idx]))
        logging.debug(f"Interpolated position: {position}")

        if array[position] == target_element:
            logging.info(f"Target {target_element} found at index {position}")
            return position
        elif array[position] < target_element:
            low_idx = position + 1
            logging.debug(f"Target greater than {array[position]}, new low={low_idx}")
        else:
            high_idx = position - 1
            logging.debug(f"Target less than {array[position]}, new high={high_idx}")

    logging.warning(f"Element {target_element} is not present in array")
    return -1

if __name__ == "__main__":
    arr = [2, 3, 4, 10, 40]
    target = 10
    result = interpolation_search_modified(arr, target)
    if result == -1:
        logging.info("Element is not present in array")
    else:
        logging.info(f"Element is present at index {result}")

    test_cases = [
        {"arr": [1, 2, 3, 4, 5], "target": 4, "expected": 3},
        {"arr": [10, 20, 30, 40, 50], "target": 25, "expected": -1},
        {"arr": [15, 15, 15, 15, 15], "target": 15, "expected": 0},  
        {"arr": [], "target": 7, "expected": -1},                   
        {"arr": [3, 6, 8, 12, 14, 17], "target": 12, "expected": 3}, 
    ]

    for idx, case in enumerate(test_cases, 1):
        arr, target, expected = case["arr"], case["target"], case["expected"]
        result = interpolation_search_modified(arr, target)
        assert result == expected, f"Test case {idx} failed: expected {expected}, got {result}"
        logging.info(f"Test case {idx} passed.")