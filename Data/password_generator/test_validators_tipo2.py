import unittest
from validator_lib import PasswordChecker

class TestPasswordChecker(unittest.TestCase):
    def setUp(self):
        self.checker = PasswordChecker()

    def test_verify_success(self):
        self.assertTrue(self.checker.verify("ValidP@ssw0rd"))

    def test_verify_length_failure(self):
        self.assertFalse(self.checker.verify("Short1!"))

    def test_verify_upper_failure(self):
        self.assertFalse(self.checker.verify("lowercase1!"))

    def test_verify_lower_failure(self):
        self.assertFalse("UPPERCASE1!")

    def test_verify_digits_failure(self):
        self.assertFalse(self.checker.verify("NoDigits!@#"))

    def test_verify_symbols_failure(self):
        self.assertFalse(self.checker.verify("NoSymbols123"))

if __name__ == '__main__':
    unittest.main()