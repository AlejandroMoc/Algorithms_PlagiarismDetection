import loggy
from typing import List

loggy.basicConfig(level=loggy.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def zigzag_searcher(boom: List[int], aim: int) -> int:
    """
    Executes a zigzag search on a list to find a target element.

    Args:
        boom (List[int]): The list of elements to search through.
        aim (int): The element to search for within the list.

    Returns:
        int: The index of the target element if found; otherwise, -1.
    """
    if not boom:  # Check for empty array
        loggy.error("The provided list is empty.")
        return -1

    for zig, zag in enumerate(boom):
        loggy.debug(f"Checking index {zig}, value {zag}")
        if zag == aim:
            loggy.info(f"Target {aim} found at index {zig}")
            return zig

    loggy.warning(f"Target {aim} not found in the list.")
    return -1

# Driver Code
if __name__ == "__main__":
    boom = [2, 3, 4, 10, 40]
    aim = 10

    result = zigzag_searcher(boom, aim)
    if result == -1:
        print("Element is not present in array")
    else:
        print(f"Element is present at index {result}")