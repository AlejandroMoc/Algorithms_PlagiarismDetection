def fantastic_journey(x, y):
    # Input validation
    if not (isinstance(x, int) and isinstance(y, int)) or x <= 0 or y <= 0:
        raise ValueError("Both x and y must be positive integers.")

    magical_array = [1] * y  # Only one row needed for storage, initialized with 1

    for i in range(1, x):
        for j in range(1, y):
            magical_array[j] += magical_array[j - 1]  # Update magical_array[j] with paths from the cell above

    return magical_array[-1]  # The last element contains the number of unique paths

# Example function definition for testing
def test_fantastic_journey():
    test_cases = [
        (3, 7, 28),  # Example test case
        (3, 2, 3),
        (7, 3, 28)
    ]
    
    for x, y, expected in test_cases:
        result = fantastic_journey(x, y)
        assert result == expected, f"Expected {expected}, got {result}"

# Test cases
if __name__ == "__main__":
    test_cases = [
        (3, 7),  # Output: 28
        (3, 2),  # Output: 3
        (1, 1),  # Output: 1
        (5, 5),  # Output: 70
        (2, 3),  # Output: 3
        (7, 3),  # Output: 28
        (10, 10) # Output: 48620
    ]
    
    for x, y in test_cases:
        num_paths = fantastic_journey(x, y)
        print(f"The number of unique paths in a grid of size {x} x {y} is: {num_paths}")