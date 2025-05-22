from dataclasses import dataclass
import logging

@dataclass
class SubarrayInput:
    numbers: list

def find_max_subarray_sum(input_data: SubarrayInput):
    numbers = input_data.numbers
    try:
        if not isinstance(numbers, list) or not numbers:
            raise ValueError("Input must be a non-empty list of numbers.")
        
        max_total = max_current = numbers[0]

        idx = 1
        while idx < len(numbers):
            max_current = max(numbers[idx], max_current + numbers[idx])
            max_total = max(max_total, max_current)
            idx += 1

        return max_total
    except Exception as exc:
        logging.error(f"Error in find_max_subarray_sum function: {exc}")
        return None

# Test cases
def test_max_subarray_sum():
    test_data = [
        (SubarrayInput([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6),
        (SubarrayInput([1]), 1),
        (SubarrayInput([0, -1, 2, -3, 4]), 4),
    ]
    
    for input_data, expected_value in test_data:
        result = find_max_subarray_sum(input_data)
        assert result == expected_value, f"Expected {expected_value}, got {result}"

if __name__ == "__main__":
    test_max_subarray_sum()