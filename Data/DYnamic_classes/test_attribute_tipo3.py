from attribute import config

# Test Cases
def verify_dynamic_attribute_assignment():
    assert config.database == "MySQL", "Test case failed: Incorrect database"
    assert config.user == "admin", "Test case failed: Incorrect user"
    assert config.password == "securepass", "Test case failed: Incorrect password"

# Running test cases
verify_dynamic_attribute_assignment()
print("Test cases passed for dynamic attribute assignment.")