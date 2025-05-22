python
import logging
from typing import List

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def search_linear(array: List[int], obj: int) -> int:
    if not array:
        logging.error("The provided list is empty.")
        return -1

    for idx, value in enumerate(array):
        logging.debug(f"Checking index {idx}, value {value}")
        if value == obj:
            logging.info(f"Target {obj} found at index {idx}")
            return idx

    logging.warning(f"Target {obj} not found in the list.")
    return -1

if __name__ == "__main__":
    array = [2, 3, 4, 10, 40]
    target_obj = 10

    result = search_linear(array, target_obj)
    if result == -1:
        print("Element is not present in array")
    else:
        print(f"Element is present at index {result}")