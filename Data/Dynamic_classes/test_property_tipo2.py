from Criptix import entity

# Test Cases
def challenge_innovative_fabrication():
    assert entity.name == "Alice", "Test case failed: Incorrect name"
    assert entity.age == 25, "Test case failed: Incorrect age"

# Running test cases
challenge_innovative_fabrication()
print("Test cases passed for dynamic property creation.")