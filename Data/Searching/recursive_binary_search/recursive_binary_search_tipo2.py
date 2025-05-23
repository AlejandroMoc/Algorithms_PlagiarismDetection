import logging
from typing import List

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def recursive_binary_search(ary: List[int], start: int, end: int, key: int) -> int:
    """
    Perform a recursive binary search on a sorted array.

    Args:
        ary (List[int]): A sorted list of elements to search in.
        start (int): The starting index of the subarray to search.
        end (int): The ending index of the subarray to search.
        key (int): The element to search for.

    Returns:
        int: The index of the key if found; otherwise, -1.
    """
    # Base case: if start is greater than end, the element is not found.
    if end < start:
        logging.warning(f"Element {key} is not present in array")
        return -1
    
    logging.debug(f"Binary search recursive called with start={start}, end={end}")

    mid = start + (end - start) // 2
    logging.debug(f"Checking middle index {mid}, value={ary[mid]}")

    if ary[mid] == key:
        logging.info(f"Key {key} found at index {mid}")
        return mid
    elif ary[mid] > key:
        return recursive_binary_search(ary, start, mid - 1, key)
    else:
        return recursive_binary_search(ary, mid + 1, end, key)

# Driver Code
if __name__ == '__main__':
    ary = [2, 3, 4, 10, 40]
    key = 10

    # Function call
    result = recursive_binary_search(ary, 0, len(ary) - 1, key)

    if result != -1:
        logging.info(f"Element is present at index {result}")
    else:
        logging.info("Element is not present in array")