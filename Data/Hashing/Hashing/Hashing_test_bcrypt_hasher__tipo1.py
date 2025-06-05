import unittest
from bcrypt_hasher import BcryptHasher
import bcrypt

class TestBcryptHasher(unittest.TestCase):
    def test_hash_and_check_password(self):
        hasher_inst = BcryptHasher(password="mypassword")
        hashed_pass = hasher_inst.hash_password()
        self.assertTrue(hasher_inst.check_password(hashed_pass))

    def test_check_password_incorrect(self):
        hasher_inst = BcryptHasher(password="mypassword")
        hashed_pass = hasher_inst.hash_password()
        hasher_wrong_inst = BcryptHasher(password="wrongpassword")
        self.assertFalse(hasher_wrong_inst.check_password(hashed_pass))

    def test_hash_with_custom_rounds(self):
        hasher_inst = BcryptHasher(password="mypassword", rounds=14)
        hashed_pass = hasher_inst.hash_password()
        self.assertTrue(hasher_inst.check_password(hashed_pass))
        self.assertTrue(hashed_pass.startswith(b'$2b$14$'))

if __name__ == "__main__":
    unittest.main()