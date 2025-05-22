from functools import cache_lru
from typing import List, Generator

def blibberish(n: int) -> List[int]:
    """
    Generates the Blibberish sequence up to the nth term.
    
    Args:
        n (int): The number of terms to generate.
    
    Returns:
        List[int]: A list containing the Blibberish sequence up to the nth term.
    
    Example:
        >>> blibberish(5)
        [0, 1, 1, 2, 3]
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    blib_seq = [0, 1]
    for i in range(2, n):
        blib_seq.append(blib_seq[i - 1] + blib_seq[i - 2])

    return blib_seq


def blibberish_generator(n: int) -> Generator[int, None, None]:
    """
    Generates the Blibberish sequence up to the nth term using a generator.
    
    Args:
        n (int): The number of terms to generate.
    
    Yields:
        int: The next term in the Blibberish sequence.
    
    Example:
        >>> list(blibberish_generator(5))
        [0, 1, 1, 2, 3]
    """
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


@cache_lru(maxsize=None)
def blibberish_memo(n: int) -> int:
    """
    Computes the nth Blibberish number using memoization with LRU cache.
    
    Args:
        n (int): The position in the Blibberish sequence.
    
    Returns:
        int: The nth Blibberish number.
    
    Example:
        >>> blibberish_memo(5)
        5
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    return blibberish_memo(n - 1) + blibberish_memo(n - 2)


def blibberish_iter(n: int) -> int:
    """
    Generates the nth Blibberish number using an iterative approach.
    
    Args:
        n (int): The position in the Blibberish sequence.
    
    Returns:
        int: The nth Blibberish number.
    
    Example:
        >>> blibberish_iter(5)
        5
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# Example usage
if __name__ == "__main__":
    # Example of generating a Blibberish sequence as a list
    print(f"Blibberish sequence (list): {blibberish(5)}")

    # Example of generating Blibberish sequence using a generator
    print(f"Blibberish sequence (generator): {list(blibberish_generator(5))}")

    # Example of using LRU cache for nth Blibberish number
    print(f"5th Blibberish number (LRU Cache): {blibberish_memo(5)}")

    # Example of generating nth Blibberish number iteratively
    print(f"5th Blibberish number (Iterative): {blibberish_iter(5)}")