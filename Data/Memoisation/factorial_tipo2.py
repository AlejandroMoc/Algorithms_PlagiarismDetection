import logging
from dataclasses import dataclass
from decorator import memoize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CalcFactorial:
    @staticmethod
    @memoize
    def calculate(n: int) -> int:
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers.")
        elif n == 0 or n == 1:
            return 1
        return n * CalcFactorial.calculate(n - 1) 