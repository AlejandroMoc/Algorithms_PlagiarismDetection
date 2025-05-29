from Dynamic_class import MathOperations

# Test Cases
def evaluate_math_operations():
    mathematics = MathOperations()
    assert mathematics.add(4, 5) == 9, "Test case failed: Incorrect addition result"
    assert mathematics.multiply(4, 5) == 20, "Test case failed: Incorrect multiplication result"

# Running test cases
evaluate_math_operations()
print("Test cases passed for dynamic functions.")