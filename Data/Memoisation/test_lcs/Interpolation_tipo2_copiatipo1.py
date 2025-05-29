import loggy
from typing import List, Optional

# Configure logging to show detailed information during search
loggy.basicConfig(
    level=loggy.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def unorthodox_search(arrival: List[int], objective: int) -> int:
    """
    Unorthodox search algorithm for sorted, uniformly distributed arrays.
    
    Args:
        arrival (List[int]): Sorted list of integers to search through.
        objective (int): The element to search for in the list.
        
    Returns:
        int: The index of the objective if found, else -1.
    """
    loggy.info("Starting unorthodox search")
    if not arrival:
        loggy.warning("Empty array provided.")
        return -1
    
    low, high = 0, len(arrival) - 1
    
    # While range is valid and objective is within the bounds
    while low <= high and arrival[low] <= objective <= arrival[high]:
        # Check for zero division edge case
        if arrival[high] == arrival[low]:
            if arrival[low] == objective:
                loggy.info(f"Objective {objective} found at index {low}")
                return low
            loggy.warning(f"Objective {objective} not found in the array.")
            return -1

        # Estimate position of the objective based on unorthodox method
        pos = low + ((objective - arrival[low]) * (high - low) // (arrival[high] - arrival[low]))
        loggy.debug(f"Unorthodox position: {pos}")

        # Compare the objective with the element at unorthodox position
        if arrival[pos] == objective:
            loggy.info(f"Objective {objective} found at index {pos}")
            return pos
        elif arrival[pos] < objective:
            low = pos + 1
            loggy.debug(f"Objective greater than {arrival[pos]}, new low={low}")
        else:
            high = pos - 1
            loggy.debug(f"Objective less than {arrival[pos]}, new high={high}")

    loggy.warning(f"Element {objective} is not present in array")
    return -1

# Driver Code and Additional Test Cases
if __name__ == "__main__":
    # Initial test case
    arrival = [2, 3, 4, 10, 40]
    objective = 10
    outcome = unorthodox_search(arrival, objective)
    if outcome == -1:
        loggy.info("Element is not present in array")
    else:
        loggy.info(f"Element is present at index {outcome}")

    # Additional test cases
    scenarios = [
        {"arrival": [1, 2, 3, 4, 5], "objective": 4, "expected": 3},
        {"arrival": [10, 20, 30, 40, 50], "objective": 25, "expected": -1},
        {"arrival": [15, 15, 15, 15, 15], "objective": 15, "expected": 0},  # Test for all identical elements
        {"arrival": [], "objective": 7, "expected": -1},                   # Test for empty array
        {"arrival": [3, 6, 8, 12, 14, 17], "objective": 12, "expected": 3}, # Standard case
    ]

    for idx, case in enumerate(scenarios, 1):
        arrival, objective, expected = case["arrival"], case["objective"], case["expected"]
        outcome = unorthodox_search(arrival, objective)
        assert outcome == expected, f"Test case {idx} failed: expected {expected}, got {outcome}"
        loggy.info(f"Test case {idx} passed.")