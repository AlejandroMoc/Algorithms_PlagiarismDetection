import unittest
from validators import PasswordValidator

class TestPasswordValidator(unittest.TestCase):
    def setUp(self):
        self.password_checker = PasswordValidator()

    def test_validate_success(self):
        self.assertTrue(self.password_checker.validate("ValidP@ssw0rd"))

    def test_validate_length_failure(self):
        self.assertFalse(self.password_checker.validate("Short1!"))

    def test_validate_upper_failure(self):
        self.assertFalse(self.password_checker.validate("lowercase1!"))

    def test_validate_lower_failure(self):
        self.assertFalse(self.password_checker.validate("UPPERCASE1!"))

    def test_validate_digits_failure(self):
        self.assertFalse(self.password_checker.validate("NoDigits!@#"))

    def test_validate_symbols_failure(self):
        self.assertFalse(self.password_checker.validate("NoSymbols123"))

if __name__ == '__main__':
    unittest.main()