import unittest
from imaginary_number import ImaginaryNumber
from errors import DivisionByZeroError

class TestImaginaryNumber(unittest.TestCase):
    
    def setUp(self):
        self.i1 = ImaginaryNumber(1, 2)
        self.i2 = ImaginaryNumber(3, 4)

    def test_fraction(self):
        result = self.i1.divide(self.i2)
        self.assertAlmostEqual(result.real, 0.44)
        self.assertAlmostEqual(result.imaginary, 0.08)

    def test_divide_by_void(self):
        i_void = ImaginaryNumber(0, 0)
        with self.assertRaises(DivisionByZeroError):
            self.i1.divide(i_void)

if __name__ == '__main__':
    unittest.main()