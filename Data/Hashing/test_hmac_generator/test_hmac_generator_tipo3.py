import unittest
from hmac_generator import HMACGenerator
import hmac
import hashlib

class TestHMACGenerator(unittest.TestCase):
    def test_generate_hmac_sha256(self):
        generator_instance = HMACGenerator(key=b'secret_key', message="Test Message", digestmod='sha256')
        expected_result = hmac.new(b'secret_key', b'Test Message', hashlib.sha256).hexdigest()
        self.assertEqual(generator_instance.generate_hmac(), expected_result)

    def test_generate_hmac_sha1(self):
        generator_instance = HMACGenerator(key=b'secret_key', message="Test Message", digestmod='sha1')
        expected_result = hmac.new(b'secret_key', b'Test Message', hashlib.sha1).hexdigest()
        self.assertEqual(generator_instance.generate_hmac(), expected_result)

    def test_invalid_digestmod(self):
        generator_instance = HMACGenerator(key=b'secret_key', message="Test Message", digestmod='invalid')
        with self.assertRaises(AttributeError):
            generator_instance.generate_hmac()

if __name__ == "__main__":
    unittest.main()