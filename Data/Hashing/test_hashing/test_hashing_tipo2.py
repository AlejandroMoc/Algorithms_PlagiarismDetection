import hashlib
import unittest
from Cryptic import CipherTable

class TestCipherTable(unittest.TestCase):
    def setUp(self):
        self.ct = CipherTable(size=5)

    def test_encrypt_and_decrypt(self):
        self.ct.encrypt(10, 'Secret1')
        self.ct.encrypt(15, 'Secret2')
        self.assertEqual(self.ct.decrypt(10), 'Secret1')
        self.assertEqual(self.ct.decrypt(15), 'Secret2')
        self.assertIsNone(self.ct.decrypt(20))

    def test_update_secret(self):
        self.ct.encrypt(10, 'Secret1')
        self.ct.encrypt(10, 'NewSecret')
        self.assertEqual(self.ct.decrypt(10), 'NewSecret')

    def test_remove_key(self):
        self.ct.encrypt(10, 'Secret1')
        self.assertTrue(self.ct.remove(10))
        self.assertIsNone(self.ct.decrypt(10))
        self.assertFalse(self.ct.remove(10))

    def test_resize_table(self):
        for i in range(10):
            self.ct.encrypt(i, f'Secret{i}')
        self.assertEqual(self.ct.size, 20)
        self.assertEqual(self.ct.decrypt(5), 'Secret5')
        self.assertEqual(self.ct.decrypt(9), 'Secret9')

    def test_data_hash(self):
        hash_value = CipherTable.data_hash("example", 10)
        self.assertEqual(hash_value, sum(ord(char) for char in "example") % 10)

    def test_detect_collisions(self):
        collisions = CipherTable.detect_collisions([1, 6, 11, 21], 5)
        self.assertEqual(collisions, [6, 11, 21])

    def test_sha256_encrypt(self):
        result = CipherTable.sha256_encrypt("Hello, World!")
        expected = hashlib.sha256("Hello, World!".encode()).hexdigest()
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()