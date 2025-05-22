import unittest
import os
from pbkdf2_hasher import PBKDF2Hasher
import hashlib

class TestPBKDF2Hasher(unittest.TestCase):
    def test_hash_password_default(self):
        salt_var = b'salt123456789012'
        hasher_var = PBKDF2Hasher(password="password", salt=salt_var)
        expected_var = hashlib.pbkdf2_hmac('sha256', b'password', salt_var, 100000, 32)
        self.assertEqual(hasher_var.hash_password(), expected_var)

    def test_hash_password_custom_iterations(self):
        salt_var = b'salt123456789012'
        hasher_var = PBKDF2Hasher(password="password", salt=salt_var, iterations=200000)
        expected_var = hashlib.pbkdf2_hmac('sha256', b'password', salt_var, 200000, 32)
        self.assertEqual(hasher_var.hash_password(), expected_var)

    def test_hash_password_invalid_salt(self):
        with self.assertRaises(AttributeError):
            PBKDF2Hasher(password="password", salt="not_bytes").hash_password()

if __name__ == "__main__":
    unittest.main()