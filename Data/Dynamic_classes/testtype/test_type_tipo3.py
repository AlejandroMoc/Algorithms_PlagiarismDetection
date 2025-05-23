from type import DynamicPerson

# Test Cases
def verify_dynamic_instance_creation():
    person = DynamicPerson()
    assert person.name == "John Doe", "Test case failed: Incorrect name"
    assert person.age == 30, "Test case failed: Incorrect age"
    assert person.greet() == "Hello, I am John Doe and I am 30 years old.", "Test case failed: Incorrect greeting"

# Running test case
verify_dynamic_instance_creation()
print("Test case passed for dynamic instance creation.")