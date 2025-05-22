import unittest
from math_operations import MathOper

class TestMathOper(unittest.TestCase):
    
    def test_oper_standard_cases(self):
        self.assertEqual(MathOper.factorial(0), 1)
        self.assertEqual(MathOper.factorial(1), 1)
        self.assertEqual(MathOper.factorial(5), 120)
        self.assertEqual(MathOper.factorial(10), 3628800)
    
    def test_oper_invalid_input(self):
        with self.assertRaises(ValueError):
            MathOper.factorial(-1)
        with self.assertRaises(ValueError):
            MathOper.factorial(-10)

if __name__ == "__main__":
    unittest.main()