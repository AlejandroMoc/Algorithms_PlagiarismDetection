import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class ShuffleSorter:
    sequence: list

    def perform_sorting(self) -> list:
        if len(self.sequence) > 1:
            mid = len(self.sequence) // 2
            left_segment = self.sequence[:mid]
            right_segment = self.sequence[mid:]

            left_sorted = ShuffleSorter(left_segment).perform_sorting()
            right_sorted = ShuffleSorter(right_segment).perform_sorting()

            return self._shuffle(left_sorted, right_sorted)
        else:
            return self.sequence

    def _shuffle(self, left, right):
        shuffled_sequence = []
        while left and right:
            if left[0] < right[0]:
                shuffled_sequence.append(left.pop(0))
            else:
                shuffled_sequence.append(right.pop(0))
        shuffled_sequence.extend(left or right)
        return shuffled_sequence

# Sample usage
if __name__ == "__main__":
    ss_instance = ShuffleSorter([34, 7, 23, 32, 5, 62])
    print(f"Shuffled sequence: {ss_instance.perform_sorting()}")  # Output: [5, 7, 23, 32, 34, 62] 