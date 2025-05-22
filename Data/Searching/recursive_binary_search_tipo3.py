import logging
from typing import List

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def recursive_binary_search(arr: List[int], low: int, high: int, target: int) -> int:
    if high < low:
        logging.warning(f"Element {target} is not present in array")
        return -1
    
    logging.debug(f"Recursive binary search called with low={low}, high={high}")

    mid = low + (high - low) // 2
    logging.debug(f"Checking middle index {mid}, value={arr[mid]}")

    if arr[mid] == target:
        logging.info(f"Target {target} found at index {mid}")
        return mid
    elif arr[mid] > target:
        return recursive_binary_search(arr, low, mid - 1, target)
    else:
        return recursive_binary_search(arr, mid + 1, high, target)

if __name__ == '__main__':
    arr = [2, 3, 4, 10, 40]
    target = 10

    result = recursive_binary_search(arr, 0, len(arr) - 1, target)

    if result != -1:
        logging.info(f"Element is present at index {result}")
    else:
        logging.info("Element is not present in array")