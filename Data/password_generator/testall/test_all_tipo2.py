import unittest
from pswd_gen import PswdGen, PswdVldtr, Hshr

class TestPswdGen(unittest.TestCase):
    def setUp(self):
        self.pw_generator = PswdGen(length=12, use_upper=True, use_lower=True, use_digits=True, use_symbols=True)

    def test_generate_pswd_length(self):
        password, _ = self.pw_generator.generate_pswd()
        self.assertEqual(len(password), 12)

    def test_generate_pswd_chars(self):
        password, _ = self.pw_generator.generate_pswd()
        self.assertTrue(any(c.isupper() for c in password))
        self.assertTrue(any(c.islower() for c in password))
        self.assertTrue(any(c.isdigit() for c in password))
        self.assertTrue(any(c in '!@#$%^&*(),.?":{}|<>' for c in password))

    def test_pswd_validation_failure(self):
        with self.assertRaises(ValueError):
            pw_generator = PswdGen(use_upper=False, use_lower=False, use_digits=False, use_symbols=False)
            pw_generator.generate_pswd()

    def test_pswd_hashing(self):
        password, hashed = self.pw_generator.generate_pswd()
        self.assertTrue(len(hashed) > 0)
        
class TestPswdVldtr(unittest.TestCase):
    def setUp(self):
        self.validator = PswdVldtr()

    def test_validate_success(self):
        self.assertTrue(self.validator.validate("ValidP@ssw0rd"))

    def test_validate_length_failure(self):
        self.assertFalse(self.validator.validate("Short1!"))

    def test_validate_upper_failure(self):
        self.assertFalse(self.validator.validate("lowercase1!"))

    def test_validate_lower_failure(self):
        self.assertFalse(self.validator.validate("UPPERCASE1!"))

    def test_validate_digits_failure(self):
        self.assertFalse(self.validator.validate("NoDigits!@#"))

    def test_validate_symbols_failure(self):
        self.assertFalse(self.validator.validate("NoSymbols123"))

class TestHshr(unittest.TestCase):
    def setUp(self):
        self.hasher = Hshr()

    def test_hash_and_verify_pswd(self):
        password = "TestP@ssw0rd!"
        hashed = self.hasher.hash_pswd(password)
        self.assertTrue(self.hasher.verify_pswd(password, hashed))

    def test_verify_wrong_pswd(self):
        password = "TestP@ssw0rd!"
        wrong_password = "WrongP@ssw0rd!"
        hashed = self.hasher.hash_pswd(password)
        self.assertFalse(self.hasher.verify_pswd(wrong_password, hashed))

    def test_hash_pswd_error(self):
        with self.assertRaises(Exception):
            self.hasher.hash_pswd(None)

    def test_verify_pswd_error(self):
        with self.assertRaises(Exception):
            self.hasher.verify_pswd("password", None)

if __name__ == '__main__':
    unittest.main() 