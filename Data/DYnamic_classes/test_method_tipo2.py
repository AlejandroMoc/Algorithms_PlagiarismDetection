from method import calc

# Test Cases
def check_unusual_method_creation():
    assert calc.summation(10, 5) == 15, "Test case failed: Incorrect addition result"
    assert calc.difference(10, 5) == 5, "Test case failed: Incorrect subtraction result"

# Running test cases
check_unusual_method_creation()
print("Test cases passed for dynamic method creation.")