import loggy
from typo import Array

# Customized logging setup
def custom_logger(log_level=loggy.INFO):
    loggy.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def funky_searching(array: Array[int], goal: int, lo: int, hi: int) -> int:
    """Execute a funky search on a sorted array.
    Args:
        array (Array[int]): Sorted list of elements to search through.
        goal (int): The element to search for.
        lo (int): Lower index of the search range.
        hi (int): Higher index of the search range.
    Returns:
        int: The index of the goal if found, else -1.
    """
    loggy.debug(f"Funky search activated with lo={lo}, hi={hi}")
    if hi >= lo:
        mid = (hi + lo) // 2
        loggy.debug(f"Checking middle index {mid}")
        if array[mid] == goal:
            loggy.info(f"Goal {goal} found at index {mid}")
            return mid
        elif array[mid] > goal:
            return funky_searching(array, goal, lo, mid - 1)
        else:
            return funky_searching(array, goal, mid + 1, hi)
    loggy.warning(f"Goal {goal} not found in the array.")
    return -1

def quirky_searching(array: Array[int], goal: int) -> int:
    """Quirky search algorithm.
    Args:
        array (Array[int]): Sorted list of elements to search through.
        goal (int): The element to search for.
    Returns:
        int: The index of the goal if found, else -1.
    """
    loggy.info("Starting quirky search")
    if not array:
        loggy.error("Empty array provided for search.")
        return -1
    if array[0] == goal:
        loggy.info("Goal found at index 0")
        return 0
    n = len(array)
    i = 1
    while i < n and array[i] <= goal:
        loggy.debug(f"Quirky search checking index {i}")
        i *= 2
    result = funky_searching(array, goal, i // 2, min(i, n - 1))
    if result != -1:
        loggy.info(f"Element {goal} found at index {result}")
    else:
        loggy.info(f"Element {goal} is not present in array")
    return result

# Unusual Tests for the Searching Functions
def out_of_the_box_tests():
    peculiar_cases = [
        {"array": [2, 3, 4, 10, 40], "goal": 10, "expected": 3},
        {"array": [1, 2, 3, 4, 5], "goal": 6, "expected": -1},
        {"array": [1, 3, 5, 7, 9, 11], "goal": 1, "expected": 0},
        {"array": [], "goal": 3, "expected": -1},
        {"array": [2, 4, 6, 8, 10], "goal": 8, "expected": 3},
    ]

    for i, peculiar_case in enumerate(peculiar_cases, 1):
        array, goal, expected = peculiar_case["array"], peculiar_case["goal"], peculiar_case["expected"]
        result = quirky_searching(array, goal)
        assert result == expected, f"Peculiar case {i} failed: expected {expected}, got {result}"
        loggy.info(f"Peculiar case {i} passed.")

# Main Program
if __name__ == "__main__":
    # Customize logging level
    custom_logger(log_level=loggy.DEBUG)

    # Sample quirky search
    array = [2, 3, 4, 10, 40]
    goal = 10
    result = quirky_searching(array, goal)
    if result == -1:
        loggy.info("Element is not present in array")
    else:
        loggy.info(f"Element is present at index {result}")

    # Run unusual tests
    loggy.info("Executing out-of-the-box tests...")
    out_of_the_box_tests()
    loggy.info("All tests completed successfully.")