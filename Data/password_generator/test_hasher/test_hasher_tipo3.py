import unittest
from hasher import Hasher

class TestHasher(unittest.TestCase):
    def setUp(self):
        self.h = Hasher()

    def test_hash_and_verify_password(self):
        p = "TestP@ssw0rd!"
        h = self.h.hash_password(p)
        self.assertTrue(self.h.verify_password(p, h))

    def test_verify_wrong_password(self):
        p = "TestP@ssw0rd!"
        w = "WrongP@ssw0rd!"
        h = self.h.hash_password(p)
        self.assertFalse(self.h.verify_password(w, h))

    def test_hash_password_error(self):
        with self.assertRaises(Exception):
            self.h.hash_password(None)

    def test_verify_password_error(self):
        with self.assertRaises(Exception):
            self.h.verify_password("password", None)

if __name__ == '__main__':
    unittest.main() 