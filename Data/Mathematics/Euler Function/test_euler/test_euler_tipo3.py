import unittest
from euler import euler_phi_function, show_coprimes

class TestEulerFunction(unittest.TestCase):
    
    def test_phi_value(self):
        self.assertEqual(euler_phi_function(1)[0], 1)
        self.assertEqual(euler_phi_function(6)[0], 2)
        self.assertEqual(euler_phi_function(9)[0], 6)
        self.assertEqual(euler_phi_function(10)[0], 4)

    def test_coprimes_list(self):
        self.assertEqual(euler_phi_function(9)[1], [1, 2, 4, 5, 7, 8])
        self.assertEqual(euler_phi_function(10)[1], [1, 3, 7, 9])
        self.assertEqual(euler_phi_function(6)[1], [1, 5])
        self.assertEqual(euler_phi_function(1)[1], [1])

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            euler_phi_function(0)
        with self.assertRaises(ValueError):
            euler_phi_function(-5)

    def test_show_coprimes(self):
        try:
            show_coprimes(10, [1, 3, 7, 9])
        except Exception as e:
            self.fail(f"show_coprimes raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()