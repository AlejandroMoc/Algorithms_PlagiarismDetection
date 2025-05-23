import logging
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fibonacci_search_algo(lst: List[int], tgt: int) -> int:
    logging.info("Starting Fibonacci search")
    if not lst:
        logging.warning("Empty list provided.")
        return -1

    n = len(lst)
    fib_mm2, fib_mm1, fib_m = 0, 1, 1

    while fib_m < n:
        fib_mm2, fib_mm1, fib_m = fib_mm1, fib_m, fib_mm2 + fib_mm1

    offset = -1

    while fib_m > 1:
        i = min(offset + fib_mm2, n - 1)
        logging.debug(f"Checking index {i}")

        if lst[i] < tgt:
            fib_m, fib_mm1, fib_mm2, offset = fib_mm1, fib_mm2, fib_m - fib_mm1, i
            logging.debug(f"Element {lst[i]} is less than target. New offset: {offset}")
        elif lst[i] > tgt:
            fib_m, fib_mm1, fib_mm2 = fib_mm2, fib_mm1 - fib_mm2, fib_m - fib_mm1
            logging.debug(f"Element {lst[i]} is greater than target.")
        else:
            logging.info(f"Target {tgt} found at index {i}")
            return i

    if fib_mm1 and offset + 1 < n and lst[offset + 1] == tgt:
        logging.info(f"Target {tgt} found at index {offset + 1}")
        return offset + 1

    logging.warning(f"Target {tgt} not found in the list.")
    return -1

if __name__ == "__main__":
    arr = [2, 3, 4, 10, 40]
    target = 10
    result = fibonacci_search_algo(arr, target)
    if result == -1:
        print("Element is not present in list")
    else:
        print("Element is present at index", result)

    test_cases = [
        {"lst": [1, 3, 5, 7, 9], "tgt": 7, "expected": 3},
        {"lst": [1, 3, 5, 7, 9], "tgt": 6, "expected": -1},
        {"lst": [], "tgt": 10, "expected": -1},
        {"lst": [2, 4, 6, 8, 10, 12], "tgt": 8, "expected": 3},
        {"lst": [1, 3, 5, 7, 9, 11], "tgt": 1, "expected": 0}
    ]

    for idx, case in enumerate(test_cases, 1):
        lst, tgt, expected = case["lst"], case["tgt"], case["expected"]
        result = fibonacci_search_algo(lst, tgt)
        assert result == expected, f"Test case {idx} failed: expected {expected}, got {result}"
        logging.info(f"Test case {idx} passed.")