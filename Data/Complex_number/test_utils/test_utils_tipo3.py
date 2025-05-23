import unittest
from utilities import read_complex

class TestUtilities(unittest.TestCase):

    def test_read_valid_positive(self):
        c = read_complex("3 + 4i")
        self.assertEqual(c.real_part, 3)
        self.assertEqual(c.imaginary_part, 4)

    def test_read_valid_negative(self):
        c = read_complex("5 - 6i")
        self.assertEqual(c.real_part, 5)
        self.assertEqual(c.imaginary_part, -6)

    def test_read_invalid_format(self):
        with self.assertRaises(ValueError):
            read_complex("invalid")

    def test_read_missing_imaginary(self):
        with self.assertRaises(ValueError):
            read_complex("3 + i")

    def test_read_missing_real(self):
        with self.assertRaises(ValueError):
            read_complex("+ 4i")

if __name__ == '__main__':
    unittest.main()