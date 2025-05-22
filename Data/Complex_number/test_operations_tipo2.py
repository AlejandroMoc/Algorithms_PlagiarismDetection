import unittest
from funky_number import FunkyNumber
from math_ops import funky_power

class FunkyTestSuite(unittest.TestCase):
    # ... existing tests ...

    def test_funky_power_null_exponent(self):
        f = FunkyNumber(2, 3)
        result = funky_power(f, 0)
        self.assertEqual(result.real, 1)
        self.assertEqual(result.imaginary, 0)

    def test_funky_power_negative_exponent(self):
        f = FunkyNumber(1, -1)
        with self.assertRaises(ValueError):
            funky_power(f, -2)
    # ... existing code ... 