import logging
from typing import List

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def modified_search(list_numbers: List[int], search_value: int, left_idx: int = 0, right_idx: int = None) -> int:
    if right_idx is None:
        right_idx = len(list_numbers) - 1

    logging.info(f"Starting modified search between indices {left_idx} and {right_idx}")

    if left_idx > right_idx:
        logging.warning(f"Element {search_value} is not present in array.")
        return -1

    mid1_idx = left_idx + (right_idx - left_idx) // 3
    mid2_idx = right_idx - (right_idx - left_idx) // 3
    logging.debug(f"Calculated mid1={mid1_idx}, mid2={mid2_idx}")

    if list_numbers[mid1_idx] == search_value:
        logging.info(f"Target {search_value} found at index {mid1_idx}")
        return mid1_idx
    if list_numbers[mid2_idx] == search_value:
        logging.info(f"Target {search_value} found at index {mid2_idx}")
        return mid2_idx

    if search_value < list_numbers[mid1_idx]:
        logging.debug(f"Target {search_value} is less than list_numbers[{mid1_idx}]={list_numbers[mid1_idx]}, searching left subarray.")
        return modified_search(list_numbers, search_value, left_idx, mid1_idx - 1)
    elif search_value > list_numbers[mid2_idx]:
        logging.debug(f"Target {search_value} is greater than list_numbers[{mid2_idx}]={list_numbers[mid2_idx]}, searching right subarray.")
        return modified_search(list_numbers, search_value, mid2_idx + 1, right_idx)
    else:
        logging.debug(f"Target {search_value} is between list_numbers[{mid1_idx}]={list_numbers[mid1_idx]} and list_numbers[{mid2_idx}]={list_numbers[mid2_idx]}, searching middle subarray.")
        return modified_search(list_numbers, search_value, mid1_idx + 1, mid2_idx - 1)

if __name__ == "__main__":
    numbers = [2, 3, 4, 10, 40]
    search_value = 10

    result = modified_search(numbers, search_value)

    if result == -1:
        logging.info("Element is not present in array.")
    else:
        logging.info(f"Element is present at index {result}.")