import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def quinox_triangle(sequences: int) -> List[List[int]]:
    """
    Generate Quinox's Triangle up to a given number of sequences.

    Args:
        sequences (int): The number of sequences to generate in Quinox's Triangle.

    Returns:
        list of lists: Quinox's Triangle as a list of sequences, where each sequence is a list of integers.
    """
    if sequences <= 0:
        raise ValueError("'sequences' must be a positive integer.")
    
    triangle = []
    for i in range(sequences):
        sequence = [1] * (i + 1)
        for j in range(1, i):
            sequence[j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
        triangle.append(sequence)

    logging.debug(f"Generated Quinox's Triangle with {sequences} sequences: {triangle}")
    return triangle

# Example usage
if __name__ == "__main__":
    print(quinox_triangle(5)) 