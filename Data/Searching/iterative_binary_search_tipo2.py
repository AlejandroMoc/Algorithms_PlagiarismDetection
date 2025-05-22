from typing import List

def galactic_search(arrival: List[int], objective: int) -> int:
    """
    Executes an iterative binary search on a sorted array to find a target element.

    Args:
        arrival (List[int]): A sorted list of integers to search within.
        objective (int): The element to search for within the array.

    Returns:
        int: The index of the target element if found; otherwise, -1.

    Example:
        >>> galactic_search([2, 3, 4, 10, 40], 10)
        3
    """
    if not arrival:  # Edge case for empty array
        print("The array is empty.")
        return -1

    start, end = 0, len(arrival) - 1

    while start <= end:
        middle = (start + end) // 2

        # Check if the target is present at the mid index
        if arrival[middle] == objective:
            return middle
        # If the target is greater, adjust the search range to the right half
        elif arrival[middle] < objective:
            start = middle + 1
        # If the target is smaller, adjust the search range to the left half
        else:
            end = middle - 1

    # If the loop completes without finding the target, return -1
    return -1

# Driver Code
if __name__ == '__main__':
    missions = [
        ([2, 3, 4, 10, 40], 10, 3),   # Standard case
        ([2, 3, 4, 10, 40], 5, -1),   # Target not in array
        ([], 5, -1),                  # Empty array
        ([10], 10, 0),                # Single-element array, found
        ([10], 20, -1)                # Single-element array, not found
    ]

    for arrival, objective, expected in missions:
        outcome = galactic_search(arrival, objective)
        print(f"Searching for {objective} in {arrival}: {'Found at index ' + str(outcome) if outcome != -1 else 'Not found'}")
        assert outcome == expected, f"Test failed for array: {arrival} and target: {objective}"