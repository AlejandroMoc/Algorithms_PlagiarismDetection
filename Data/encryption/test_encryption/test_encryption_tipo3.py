import unittest
from Crypto.Random import get_random_bytes
from encryption import AESCipher

class TestAESCipher(unittest.TestCase):
    def setUp(self):
        self.password = "test_passphrase"
        self.salt_bytes = get_random_bytes(16)
        self.secret_key = AESCipher.generate_key(self.password, self.salt_bytes)
        self.encryption = AESCipher(self.secret_key)

    def test_encryption_decryption(self):
        original_info = b"Hello, World!"
        encrypted_info = self.encryption.encrypt(original_info)
        decrypted_info = self.encryption.decrypt(encrypted_info)
        self.assertEqual(original_info.decode(), decrypted_info)

    def test_padding_unpadding(self):
        info = b"Hello"
        padded_info = self.encryption.pad(info)
        unpadded_info = self.encryption.unpad(padded_info)
        self.assertEqual(info, unpadded_info)

if __name__ == "__main__":
    unittest.main()