python
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class MergeSortAlgorithm:
    data: list

    def sort_data(self) -> list:
        if len(self.data) > 1:
            mid_point = len(self.data) // 2
            left_partition = self.data[:mid_point]
            right_partition = self.data[mid_point:]

            left_sorted = MergeSortAlgorithm(left_partition).sort_data()
            right_sorted = MergeSortAlgorithm(right_partition).sort_data()

            return self._merge_arrays(left_sorted, right_sorted)
        else:
            return self.data

    def _merge_arrays(self, left_array, right_array):
        sorted_result = []
        while left_array and right_array:
            if left_array[0] < right_array[0]:
                sorted_result.append(left_array.pop(0))
            else:
                sorted_result.append(right_array.pop(0))
        sorted_result.extend(left_array or right_array)
        return sorted_result

# Sample usage
if __name__ == "__main__":
    ms_algorithm = MergeSortAlgorithm([34, 7, 23, 32, 5, 62])
    print(f"Sorted array: {ms_algorithm.sort_data()}")  # Output: [5, 7, 23, 32, 34, 62]