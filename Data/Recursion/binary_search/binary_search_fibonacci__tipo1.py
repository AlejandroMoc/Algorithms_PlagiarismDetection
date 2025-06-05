import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class FibonacciSequence:
    number: int

    def calculate(self) -> int:
        if self.number < 0:
            raise ValueError("Input must be a non-negative integer")
        return self._calculate_fibonacci(self.number)

    def _calculate_fibonacci(self, current: int) -> int:
        if current in (0, 1):
            return current
        result = self._calculate_fibonacci(current - 1) + self._calculate_fibonacci(current - 2)
        logging.info(f"Fibonacci({current}) = {result}")
        return result

# Sample usage
if __name__ == "__main__":
    fibonacci_sequence = FibonacciSequence(10)
    print(f"Fibonacci of 10 is {fibonacci_sequence.calculate()}")  # Output: 55