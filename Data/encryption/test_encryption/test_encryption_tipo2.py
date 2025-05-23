import unittest
from Crypto.Random import get_random_bytes
from encryption import AESCipher

class TestXZYEncryptor(unittest.TestCase):
    def setUp(self):
        self.secret_key = "test_secret_key"
        self.salt = get_random_bytes(16)
        self.key = AESCipher.generate_key(self.secret_key, self.salt)
        self.encryptor = AESCipher(self.key)

    def test_encoding_decoding(self):
        original_text = b"Hello, World!"
        encoded_text = self.encryptor.encrypt(original_text)
        decoded_text = self.encryptor.decrypt(encoded_text)
        self.assertEqual(original_text.decode(), decoded_text)

    def test_padding_stripping(self):
        text = b"Hello"
        padded_text = self.encryptor.pad(text)
        stripped_text = self.encryptor.strip_padding(padded_text)
        self.assertEqual(text, stripped_text)

if __name__ == "__main__":
    unittest.main()