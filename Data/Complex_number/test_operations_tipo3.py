python
import unittest
from complex_number import ComplexNumber
from operations import power

class TestOperations(unittest.TestCase):
    # ... existing tests ...

    def test_power_zero_exponent(self):
        comp_num = ComplexNumber(2, 3)
        result = power(comp_num, 0)
        self.assertEqual(result.real, 1)
        self.assertEqual(result.imaginary, 0)

    def test_power_negative_exponent(self):
        comp_num = ComplexNumber(1, -1)
        with self.assertRaises(ValueError):
            power(comp_num, -2)
    # ... existing code ...