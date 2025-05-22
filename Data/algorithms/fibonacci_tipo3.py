from functools import lru_cache
from typing import List, Generator

def generate_fibonacci_sequence(n: int) -> List[int]:
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    fib_seq = [0, 1]
    i = 2
    while i < n:
        fib_seq.append(fib_seq[i - 1] + fib_seq[i - 2])
        i += 1

    return fib_seq

def generate_fibonacci_sequence_generator(n: int) -> Generator[int, None, None]:
    a, b = 0, 1
    i = 0
    while i < n:
        yield a
        a, b = b, a + b
        i += 1

@lru_cache(maxsize=None)
def calculate_fibonacci_number(n: int) -> int:
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    return calculate_fibonacci_number(n - 1) + calculate_fibonacci_number(n - 2)

def calculate_fibonacci_number_iterative(n: int) -> int:
    if n <= 0:
        return 0
    elif n == 1:
        return 1

    a, b = 0, 1
    i = 2
    while i < n + 1:
        a, b = b, a + b
        i += 1
    return b

if __name__ == "__main__":
    print(f"Fibonacci sequence (list): {generate_fibonacci_sequence(5)}")
    print(f"Fibonacci sequence (generator): {list(generate_fibonacci_sequence_generator(5))}")
    print(f"5th Fibonacci number (LRU Cache): {calculate_fibonacci_number(5)}")
    print(f"5th Fibonacci number (Iterative): {calculate_fibonacci_number_iterative(5)}")