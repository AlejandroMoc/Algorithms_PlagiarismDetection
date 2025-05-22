from typing import List

def iterative_binary_search(array: List[int], target_value: int) -> int:
    if not array:
        print("The array is empty.")
        return -1

    left, right = 0, len(array) - 1

    while left <= right:
        middle = (left + right) // 2

        if array[middle] == target_value:
            return middle
        elif array[middle] < target_value:
            left = middle + 1
        else:
            right = middle - 1

    return -1

if __name__ == '__main__':
    test_cases = [
        ([2, 3, 4, 10, 40], 10, 3),
        ([2, 3, 4, 10, 40], 5, -1),
        ([], 5, -1),
        ([10], 10, 0),
        ([10], 20, -1)
    ]

    for array, target_value, expected_index in test_cases:
        result_index = iterative_binary_search(array, target_value)
        print(f"Searching for {target_value} in {array}: {'Found at index ' + str(result_index) if result_index != -1 else 'Not found'}")
        assert result_index == expected_index, f"Test failed for array: {array} and target: {target_value}"