import unittest
from hash_generator import HashGenerator

class TestHashGenerator(unittest.TestCase):
    def test_generate_hash_normal_sequence(self):
        generator = HashGenerator(sequence="Hello, World!")
        self.assertEqual(generator.generate_hash(), "fc3ff98e8c6a0d3087d515c0473f8677")

    def test_generate_hash_empty_sequence(self):
        generator = HashGenerator(sequence="")
        self.assertEqual(generator.generate_hash(), "d41d8cd98f00b204e9800998ecf8427e")

    def test_generate_hash_unicode_sequence(self):
        generator = HashGenerator(sequence="こんにちは")
        self.assertEqual(generator.generate_hash(), "9d735278cfbdb946834416adfb5aaf6c")

if __name__ == "__main__":
    unittest.main()