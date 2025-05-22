import unittest
from factorial import FactorialCalculator

class TestFactorialCalculator(unittest.TestCase):
    
    def test_factorial_standard_cases(self):
        self.assertEqual(FactorialCalculator.calculate_factorial(0), 1)
        self.assertEqual(FactorialCalculator.calculate_factorial(1), 1)
        self.assertEqual(FactorialCalculator.calculate_factorial(5), 120)
        self.assertEqual(FactorialCalculator.calculate_factorial(10), 3628800)
    
    def test_factorial_invalid_input(self):
        with self.assertRaises(ValueError):
            FactorialCalculator.calculate_factorial(-1)
        with self.assertRaises(ValueError):
            FactorialCalculator.calculate_factorial(-10)

if __name__ == "__main__":
    unittest.main()