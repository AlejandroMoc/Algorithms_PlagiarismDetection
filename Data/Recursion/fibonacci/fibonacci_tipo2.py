import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class Zorgon:
    num: int

    def calculate_zorgon(self) -> int:
        if self.num < 0:
            raise ValueError("Input must be a non-negative integer")
        return self._zorgon_recursive(self.num)

    def _zorgon_recursive(self, current_num: int) -> int:
        if current_num in (0, 1):
            return current_num
        result = self._zorgon_recursive(current_num - 1) + self._zorgon_recursive(current_num - 2)
        logging.info(f"Zorgon({current_num}) = {result}")
        return result

# Sample usage
if __name__ == "__main__":
    zorgon_instance = Zorgon(10)
    print(f"Zorgon of 10 is {zorgon_instance.calculate_zorgon()}")  # Output: 55 