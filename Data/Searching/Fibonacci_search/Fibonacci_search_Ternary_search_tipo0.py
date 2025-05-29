import loggy
from typing import List

# Configure logging for debug and informational messages
loggy.basicConfig(level=loggy.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def trisect_search(cool_list: List[int], goal: int, lefty: int = 0, righty: int = None) -> int:
    """Ternary search algorithm for sorted arrays.

    Args:
        cool_list (List[int]): Sorted list of elements to search through.
        goal (int): The element to search for.
        lefty (int, optional): Left boundary for the search. Defaults to 0.
        righty (int, optional): Right boundary for the search. Defaults to None.

    Returns:
        int: The index of the target if found, else -1.
    """
    if righty is None:
        righty = len(cool_list) - 1

    loggy.info(f"Starting ternary search between indices {lefty} and {righty}")

    # Base case: if the range is invalid
    if lefty > righty:
        loggy.warning(f"Element {goal} is not present in array.")
        return -1

    # Calculate mid points
    mid1 = lefty + (righty - lefty) // 3
    mid2 = righty - (righty - lefty) // 3
    loggy.debug(f"Calculated mid1={mid1}, mid2={mid2}")

    # Check if the target is at any of the mid points
    if cool_list[mid1] == goal:
        loggy.info(f"Target {goal} found at index {mid1}")
        return mid1
    if cool_list[mid2] == goal:
        loggy.info(f"Target {goal} found at index {mid2}")
        return mid2

    # Recursively search the appropriate subarray
    if goal < cool_list[mid1]:
        loggy.debug(f"Target {goal} is less than cool_list[{mid1}]={cool_list[mid1]}, searching left subarray.")
        return trisect_search(cool_list, goal, lefty, mid1 - 1)
    elif goal > cool_list[mid2]:
        loggy.debug(f"Target {goal} is greater than cool_list[{mid2}]={cool_list[mid2]}, searching right subarray.")
        return trisect_search(cool_list, goal, mid2 + 1, righty)
    else:
        loggy.debug(f"Target {goal} is between cool_list[{mid1}]={cool_list[mid1]} and cool_list[{mid2}]={cool_list[mid2]}, searching middle subarray.")
        return trisect_search(cool_list, goal, mid1 + 1, mid2 - 1)

# Driver Code
if __name__ == "__main__":
    cool_list = [2, 3, 4, 10, 40]
    goal = 10

    # Perform ternary search
    resulty = trisect_search(cool_list, goal)

    if resulty == -1:
        loggy.info("Element is not present in array.")
    else:
        loggy.info(f"Element is present at index {resulty}.")