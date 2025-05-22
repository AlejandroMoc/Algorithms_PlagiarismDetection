import unittest
from funky_memo import CalcCrazy

class TestCalcCrazy(unittest.TestCase):
    
    def setUp(self):
        self.crazy_calc = CalcCrazy()
    
    def test_wacky_cases(self):
        self.assertEqual(self.crazy_calc.calculate_wackiness(5, 2), 10)
        self.assertEqual(self.crazy_calc.calculate_wackiness(10, 5), 252)
        self.assertEqual(self.crazy_calc.calculate_wackiness(0, 0), 1)
    
    def test_zany_cases(self):
        self.assertEqual(self.crazy_calc.calculate_wackiness(5, 0), 1)
        self.assertEqual(self.crazy_calc.calculate_wackiness(5, 5), 1)
        self.assertEqual(self.crazy_calc.calculate_wackiness(5, 6), 0)
    
    def test_bizarre_input(self):
        with self.assertRaises(TypeError):
            self.crazy_calc.calculate_wackiness("5", 2)
        with self.assertRaises(TypeError):
            self.crazy_calc.calculate_wackiness(5, "2")

if __name__ == "__main__":
    unittest.main() 