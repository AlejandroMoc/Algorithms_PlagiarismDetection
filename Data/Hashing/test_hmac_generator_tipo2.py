import unittest
from hashy_hmac import HashyHMAC
import hmac
import hashlib

class TestHashyHMAC(unittest.TestCase):
    def test_compute_hashy_hmac_sha256(self):
        hasher = HashyHMAC(key=b'secret_key', message="Test Message", algo='sha256')
        expected = hmac.new(b'secret_key', b'Test Message', hashlib.sha256).hexdigest()
        self.assertEqual(hasher.compute_hashy_hmac(), expected)

    def test_compute_hashy_hmac_sha1(self):
        hasher = HashyHMAC(key=b'secret_key', message="Test Message", algo='sha1')
        expected = hmac.new(b'secret_key', b'Test Message', hashlib.sha1).hexdigest()
        self.assertEqual(hasher.compute_hashy_hmac(), expected)

    def test_invalid_algo(self):
        hasher = HashyHMAC(key=b'secret_key', message="Test Message", algo='invalid')
        with self.assertRaises(AttributeError):
            hasher.compute_hashy_hmac()

if __name__ == "__main__":
    unittest.main()