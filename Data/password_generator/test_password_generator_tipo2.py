import unittest
from pw_gen import PwGen

class TestPwGen(unittest.TestCase):
    def setUp(self):
        self.pwgen = PwGen(length=12, use_upper=True, use_lower=True, use_digits=True, use_symbols=True)

    def test_generate_pw_length(self):
        password, _ = self.pwgen.generate_pw()
        self.assertEqual(len(password), 12)

    def test_generate_pw_chars(self):
        password, _ = self.pwgen.generate_pw()
        self.assertTrue(any(c.isupper() for c in password))
        self.assertTrue(any(c.islower() for c in password))
        self.assertTrue(any(c.isdigit() for c in password))
        self.assertTrue(any(c in '!@#$%^&*(),.?":{}|<>' for c in password))

    def test_pw_validation_failure(self):
        with self.assertRaises(ValueError):
            pwgen = PwGen(use_upper=False, use_lower=False, use_digits=False, use_symbols=False)
            pwgen.generate_pw()

    def test_pw_hashing(self):
        password, hashed = self.pwgen.generate_pw()
        self.assertTrue(len(hashed) > 0)

if __name__ == '__main__':
    unittest.main()