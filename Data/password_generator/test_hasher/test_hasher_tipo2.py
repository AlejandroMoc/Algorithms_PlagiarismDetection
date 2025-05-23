import unittest
from cryptographer import Cryptographer

class TestCryptographer(unittest.TestCase):
    def setUp(self):
        self.crypt = Cryptographer()

    def test_encrypt_and_decrypt_data(self):
        data = "TestData123!"
        encrypted = self.crypt.encrypt_data(data)
        self.assertTrue(self.crypt.decrypt_data(data, encrypted))

    def test_decrypt_wrong_data(self):
        data = "TestData123!"
        wrong_data = "WrongData123!"
        encrypted = self.crypt.encrypt_data(data)
        self.assertFalse(self.crypt.decrypt_data(wrong_data, encrypted))

    def test_encrypt_data_error(self):
        with self.assertRaises(Exception):
            self.crypt.encrypt_data(None)

    def test_decrypt_data_error(self):
        with self.assertRaises(Exception):
            self.crypt.decrypt_data("data", None)

if __name__ == '__main__':
    unittest.main() 