import unittest
from sha3_hasher import SHA3Hasher
import hashlib

class TestSHA3Hasher(unittest.TestCase):
    def test_hash_sha3_256(self):
        hash_generator = SHA3Hasher(data="Test Data", variant='sha3_256')
        expected_result = hashlib.sha3_256(b'Test Data').hexdigest()
        self.assertEqual(hash_generator.hash_data(), expected_result)

    def test_hash_sha3_512(self):
        hash_generator = SHA3Hasher(data="Test Data", variant='sha3_512')
        expected_result = hashlib.sha3_512(b'Test Data').hexdigest()
        self.assertEqual(hash_generator.hash_data(), expected_result)

    def test_invalid_variant(self):
        hash_generator = SHA3Hasher(data="Test Data", variant='sha3_999')
        with self.assertRaises(AttributeError):
            hash_generator.hash_data()

if __name__ == "__main__":
    unittest.main() 