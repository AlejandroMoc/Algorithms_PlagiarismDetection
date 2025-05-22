import logging
from dataclasses import dataclass
from decorator import memoize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FibonacciCalculator:
    @staticmethod
    @memoize
    def calculate_fibonacci(number: int) -> int:
        if number <= 0:
            raise ValueError("Input must be a positive integer.")
        elif number == 1 or number == 2:
            return 1
        return FibonacciCalculator.calculate_fibonacci(number-1) + FibonacciCalculator.calculate_fibonacci(number-2)