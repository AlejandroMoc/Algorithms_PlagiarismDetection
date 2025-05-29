import unittest
from crypto import decrypt_secret

class TestCrypto(unittest.TestCase):

    def test_decrypt_valid_positive(self):
        s = decrypt_secret("3 + 4i")
        self.assertEqual(s.real, 3)
        self.assertEqual(s.imaginary, 4)

    def test_decrypt_valid_negative(self):
        s = decrypt_secret("5 - 6i")
        self.assertEqual(s.real, 5)
        self.assertEqual(s.imaginary, -6)

    def test_decrypt_invalid_format(self):
        with self.assertRaises(ValueError):
            decrypt_secret("invalid")

    def test_decrypt_missing_imaginary(self):
        with self.assertRaises(ValueError):
            decrypt_secret("3 + i")

    def test_decrypt_missing_real(self):
        with self.assertRaises(ValueError):
            decrypt_secret("+ 4i")

if __name__ == '__main__':
    unittest.main()