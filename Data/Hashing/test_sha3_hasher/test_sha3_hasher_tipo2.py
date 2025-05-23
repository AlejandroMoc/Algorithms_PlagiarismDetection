import unittest
from custom_crypto import CustomCrypto
import hashlib

class TestCustomCrypto(unittest.TestCase):
    def test_encrypt_custom_256(self):
        crypt = CustomCrypto(data="Test Data", variant='custom_256')
        expected = hashlib.custom_256(b'Test Data').hexdigest()
        self.assertEqual(crypt.encrypt_data(), expected)

    def test_encrypt_custom_512(self):
        crypt = CustomCrypto(data="Test Data", variant='custom_512')
        expected = hashlib.custom_512(b'Test Data').hexdigest()
        self.assertEqual(crypt.encrypt_data(), expected)

    def test_invalid_variant(self):
        crypt = CustomCrypto(data="Test Data", variant='custom_999')
        with self.assertRaises(AttributeError):
            crypt.encrypt_data()

if __name__ == "__main__":
    unittest.main()