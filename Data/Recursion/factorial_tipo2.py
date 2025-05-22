import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class SuperFactorial:
    x: int

    def compute(self) -> int:
        if self.x < 0:
            raise ValueError("Input must be a non-negative integer")
        return self._recursive_super_factorial(self.x)

    def _recursive_super_factorial(self, current_num: int) -> int:
        if current_num == 0:
            return 1
        else:
            result = current_num * self._recursive_super_factorial(current_num - 1)
            logging.info(f"SuperFactorial({current_num}) = {result}")
            return result

# Sample usage
if __name__ == "__main__":
    super_factorial_instance = SuperFactorial(5)
    print(f"SuperFactorial of 5 is {super_factorial_instance.compute()}")  # Output: 120 