from method import calc

# Test Cases
def test_dynamic_method_creation():
    assert calc.addition(10, 5) == 15, "Test case failed: Incorrect addition result"
    assert calc.subtraction(10, 5) == 5, "Test case failed: Incorrect subtraction result"

# Running test cases
test_dynamic_method_creation()
print("Test cases passed for dynamic method creation.")