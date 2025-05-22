import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class FactorialCalculation:
    num: int

    def compute(self) -> int:
        if self.num < 0:
            raise ValueError("Input must be a non-negative integer")
        return self._recursive_factorial(self.num)

    def _recursive_factorial(self, current_num: int) -> int:
        if current_num == 0:
            return 1
        else:
            result = current_num * self._recursive_factorial(current_num - 1)
            logging.info(f"Factorial({current_num}) = {result}")
            return result

# Sample usage
if __name__ == "__main__":
    factorial_calculation = FactorialCalculation(5)
    print(f"Factorial of 5 is {factorial_calculation.compute()}")  # Output: 120