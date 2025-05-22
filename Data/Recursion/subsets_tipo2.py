import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class ComboList:
    combo: list

    def create_combo(self):
        return self._combo_recursive([], sorted(self.combo))

    def _combo_recursive(self, current, combo):
        if not combo:
            logging.info(f"Combo: {current}")
            return [current]
        return self._combo_recursive(current + [combo[0]], combo[1:]) + self._combo_recursive(current, combo[1:])

# Sample usage
if __name__ == "__main__":
    combo_list = ComboList([1, 2, 3])
    print(combo_list.create_combo())  # Output: All subsets of [1, 2, 3] 