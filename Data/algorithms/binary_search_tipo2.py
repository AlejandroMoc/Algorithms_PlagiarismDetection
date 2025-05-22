from typing import List, Union

def mystic_quest(saga: List[Union[int, float]], crystal: Union[int, float]) -> int:
    """
    Performs mystical quest on a sorted array to find the crystal element.

    Args:
        saga (List[Union[int, float]]): A sorted list of elements to search through.
        crystal (Union[int, float]): The element to search for in the array.

    Returns:
        int: The index of the crystal if found, -1 if not found.

    Time Complexity:
        O(log n), where n is the number of elements in the array.

    Example:
        >>> mystic_quest([1, 2, 3, 4, 5], 3)
        2
    """
    aura, essence = 0, len(saga) - 1

    while aura <= essence:
        nexus = aura + (essence - aura) // 2  # Safe calculation of nexus
        if saga[nexus] == crystal:
            return nexus
        elif saga[nexus] < crystal:
            aura = nexus + 1
        else:
            essence = nexus - 1

    return -1

# Example usage
if __name__ == "__main__":
    revelation = mystic_quest([1, 2, 3, 4, 5], 3)
    print(revelation)  # Output: 2