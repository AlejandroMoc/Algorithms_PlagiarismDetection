import unittest
from xozz_hasher import XozzHasher
import bcrypt

class TestXozzHasher(unittest.TestCase):
    def test_encrypt_and_verify_secret(self):
        hasher = XozzHasher(secret_key="mysupersecret")
        encrypted = hasher.encrypt_secret()
        self.assertTrue(hasher.verify_secret(encrypted))

    def test_verify_secret_incorrect(self):
        hasher = XozzHasher(secret_key="mysupersecret")
        encrypted = hasher.encrypt_secret()
        hasher_wrong = XozzHasher(secret_key="wrongsecret")
        self.assertFalse(hasher_wrong.verify_secret(encrypted))

    def test_encrypt_with_custom_iterations(self):
        hasher = XozzHasher(secret_key="mysupersecret", iterations=14)
        encrypted = hasher.encrypt_secret()
        self.assertTrue(hasher.verify_secret(encrypted))
        self.assertTrue(encrypted.startswith(b'$2b$14$'))

if __name__ == "__main__":
    unittest.main()