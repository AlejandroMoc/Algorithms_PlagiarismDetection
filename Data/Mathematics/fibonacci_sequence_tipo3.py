import logging
from typing import List, Union

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def generate_fibonacci_sequence(amount: Union[int, None] = None, maximum: Union[int, None] = None) -> List[int]:
    """
    Generate a Fibonacci sequence up to a specified count or maximum value.

    Args:
        amount (int, optional): Number of Fibonacci numbers to generate.
        maximum (int, optional): Maximum value of Fibonacci numbers.

    Returns:
        list: List of Fibonacci numbers.
    """
    if amount is None and maximum is None:
        raise ValueError("Either 'amount' or 'maximum' must be provided.")

    if amount is not None and amount <= 0:
        raise ValueError("'amount' must be a positive integer.")

    if maximum is not None and maximum < 0:
        raise ValueError("'maximum' must be a non-negative integer.")

    sequence = []
    first, second = 0, 1

    while (amount is None or len(sequence) < amount) and (maximum is None or first <= maximum):
        sequence.append(first)
        first, second = second, first + second

    logging.debug(f"Generated Fibonacci sequence: {sequence}")
    return sequence

# Example usage
if __name__ == "__main__":
    print(generate_fibonacci_sequence(amount=10))