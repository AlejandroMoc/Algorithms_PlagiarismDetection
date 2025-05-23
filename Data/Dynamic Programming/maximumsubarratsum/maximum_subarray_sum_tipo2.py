from dataclasses import dataclass
import logging

@dataclass
class SubarrayData:
    array: list

def calculate_maximum_subarray(subarray_info: SubarrayData):
    array = subarray_info.array
    try:
        if not isinstance(array, list) or not array:
            raise ValueError("Input must be a non-empty list of numbers.")
        
        max_so_far = max_ending_here = array[0]

        for num in array[1:]:
            max_ending_here = max(num, max_ending_here + num)
            max_so_far = max(max_so_far, max_ending_here)

        return max_so_far
    except Exception as e:
        logging.error(f"Error in calculate_maximum_subarray function: {e}")
        return None

# Test cases
def validate_maximum_subarray():
    test_cases = [
        (SubarrayData([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6),
        (SubarrayData([1]), 1),
        (SubarrayData([0, -1, 2, -3, 4]), 4),
    ]
    
    for subarray_info, expected_result in test_cases:
        result = calculate_maximum_subarray(subarray_info)
        assert result == expected_result, f"Expected {expected_result}, got {result}"

if __name__ == "__main__":
    validate_maximum_subarray()