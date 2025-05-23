import unittest
import os
from cryptic_hasher import CrypticHasher
import hashlib

class TestCrypticHasher(unittest.TestCase):
    def test_obfuscate_data_default(self):
        shadow = b'shadow123456789012'
        hasher = CrypticHasher(secret="secret", shadow=shadow)
        expected = hashlib.pbkdf2_hmac('sha256', b'secret', shadow, 100000, 32)
        self.assertEqual(hasher.obfuscate_data(), expected)

    def test_obfuscate_data_custom_iterations(self):
        shadow = b'shadow123456789012'
        hasher = CrypticHasher(secret="secret", shadow=shadow, iterations=200000)
        expected = hashlib.pbkdf2_hmac('sha256', b'secret', shadow, 200000, 32)
        self.assertEqual(hasher.obfuscate_data(), expected)

    def test_obfuscate_data_invalid_shadow(self):
        with self.assertRaises(AttributeError):
            CrypticHasher(secret="secret", shadow="not_bytes").obfuscate_data()

if __name__ == "__main__":
    unittest.main()