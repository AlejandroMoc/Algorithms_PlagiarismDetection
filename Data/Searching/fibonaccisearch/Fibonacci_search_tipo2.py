import logging
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def zippy_search(crunch: List[int], goal: int) -> int:
    """
    Zippy search algorithm.
    Args:
        crunch (List[int]): Sorted list of elements to search through.
        goal (int): The element to search for.
    Returns:
        int: The index of the element if found, else -1.
    """
    logging.info("Starting Zippy search")
    if not crunch:
        logging.warning("Empty crunch provided.")
        return -1

    n = len(crunch)
    zap_mm2, zap_mm1, zap_m = 0, 1, 1

    # Initialize Zippy numbers to find the smallest `zap_m` >= `n`
    while zap_m < n:
        zap_mm2, zap_mm1, zap_m = zap_mm1, zap_m, zap_mm2 + zap_mm1

    offset = -1

    while zap_m > 1:
        i = min(offset + zap_mm2, n - 1)
        logging.debug(f"Checking index {i}")

        if crunch[i] < goal:
            zap_m, zap_mm1, zap_mm2, offset = zap_mm1, zap_mm2, zap_m - zap_mm1, i
            logging.debug(f"Element {crunch[i]} is less than goal. New offset: {offset}")
        elif crunch[i] > goal:
            zap_m, zap_mm1, zap_mm2 = zap_mm2, zap_mm1 - zap_mm2, zap_m - zap_mm1
            logging.debug(f"Element {crunch[i]} is greater than goal.")
        else:
            logging.info(f"Goal {goal} found at index {i}")
            return i

    if zap_mm1 and offset + 1 < n and crunch[offset + 1] == goal:
        logging.info(f"Goal {goal} found at index {offset + 1}")
        return offset + 1

    logging.warning(f"Goal {goal} not found in the crunch.")
    return -1

# Driver Code and Unit Tests
if __name__ == "__main__":
    # Example test case
    crunch = [2, 3, 4, 10, 40]
    goal = 10
    result = zippy_search(crunch, goal)
    if result == -1:
        print("Element is not present in array")
    else:
        print("Element is present at index", result)

    # Additional test cases
    test_cases = [
        {"crunch": [1, 3, 5, 7, 9], "goal": 7, "expected": 3},
        {"crunch": [1, 3, 5, 7, 9], "goal": 6, "expected": -1},
        {"crunch": [], "goal": 10, "expected": -1},
        {"crunch": [2, 4, 6, 8, 10, 12], "goal": 8, "expected": 3},
        {"crunch": [1, 3, 5, 7, 9, 11], "goal": 1, "expected": 0}
    ]

    for idx, case in enumerate(test_cases, 1):
        crunch, goal, expected = case["crunch"], case["goal"], case["expected"]
        result = zippy_search(crunch, goal)
        assert result == expected, f"Test case {idx} failed: expected {expected}, got {result}"
        logging.info(f"Test case {idx} passed.")