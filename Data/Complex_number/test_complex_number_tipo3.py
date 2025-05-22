import unittest
from complex_number import ComplexNumber
from exceptions import DivisionByZeroError

class TestComplexNumber(unittest.TestCase):
    
    def setUp(self):
        self.complex_num1 = ComplexNumber(1, 2)
        self.complex_num2 = ComplexNumber(3, 4)

    def test_division(self):
        result = self.complex_num1.divide(self.complex_num2)
        self.assertAlmostEqual(result.real, 0.44)
        self.assertAlmostEqual(result.imaginary, 0.08)

    def test_division_by_zero(self):
        complex_zero = ComplexNumber(0, 0)
        with self.assertRaises(DivisionByZeroError):
            self.complex_num1.divide(complex_zero)

if __name__ == '__main__':
    unittest.main()