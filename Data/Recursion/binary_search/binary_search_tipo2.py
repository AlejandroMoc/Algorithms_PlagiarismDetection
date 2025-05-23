import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class ZAlgorithm:
    data_list: list
    goal: int

    def process(self) -> int:
        return self._z_recursive_algorithm(self.data_list, self.goal, 0, len(self.data_list) - 1)

    def _z_recursive_algorithm(self, array, target, left, right):
        if right >= left:
            mid = left + (right - left) // 2
            if array[mid] == target:
                logging.info(f"Goal {target} reached at position {mid}")
                return mid
            elif array[mid] > target:
                return self._z_recursive_algorithm(array, target, left, mid - 1)
            else:
                return self._z_recursive_algorithm(array, target, mid + 1, right)
        else:
            logging.info(f"Goal {target} not achieved")
            return -1

# Sample usage
if __name__ == "__main__":
    za_instance = ZAlgorithm([1, 2, 3, 4, 5], 3)
    print(f"Position of 3 is {za_instance.process()}")  # Output: 2 