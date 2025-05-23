import unittest
from ploshemo import SplendoCalculator

class TestSplendoCalculator(unittest.TestCase):
    
    def test_splendo_standard_cases(self):
        self.assertEqual(SplendoCalculator.splendo(1), 1)
        self.assertEqual(SplendoCalculator.splendo(5), 5)
        self.assertEqual(SplendoCalculator.splendo(10), 55)
    
    def test_splendo_invalid_input(self):
        with self.assertRaises(ValueError):
            SplendoCalculator.splendo(0)
        with self.assertRaises(ValueError):
            SplendoCalculator.splendo(-5)

if __name__ == "__main__":
    unittest.main()