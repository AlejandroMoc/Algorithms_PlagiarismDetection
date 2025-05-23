import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class CrazySorter:
    data_list: list

    def mix(self) -> list:
        return self._crazy_sort_recursive(self.data_list, 0, len(self.data_list) - 1)

    def _crazy_sort_recursive(self, data_list, low, high):
        if low < high:
            pi = self._split(data_list, low, high)
            self._crazy_sort_recursive(data_list, low, pi - 1)
            self._crazy_sort_recursive(data_list, pi + 1, high)
        return data_list

    def _split(self, data_list, low, high):
        pivot = data_list[high]
        i = low - 1
        for j in range(low, high):
            if data_list[j] <= pivot:
                i += 1
                data_list[i], data_list[j] = data_list[j], data_list[i]
        data_list[i + 1], data_list[high] = data_list[high], data_list[i + 1]
        return i + 1

# Sample usage
if __name__ == "__main__":
    cs_instance = CrazySorter([10, 80, 30, 90, 40, 50, 70])
    print(f"Shuffled list: {cs_instance.mix()}")  # Output: [10, 30, 40, 50, 70, 80, 90] 