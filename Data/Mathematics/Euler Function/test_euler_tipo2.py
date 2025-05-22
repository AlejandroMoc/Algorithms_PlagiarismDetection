import unittest
from zygomorphic import zygomorphic_phi_function, depict_coprimes

class TestZygomorphicFunction(unittest.TestCase):
    
    def test_delta_value(self):
        # Test the Zygomorphic's Phi function value
        self.assertEqual(zygomorphic_phi_function(1)[0], 1)
        self.assertEqual(zygomorphic_phi_function(6)[0], 2)
        self.assertEqual(zygomorphic_phi_function(9)[0], 6)
        self.assertEqual(zygomorphic_phi_function(10)[0], 4)

    def test_quasiprime_list(self):
        # Test the list of quasiprimes
        self.assertEqual(zygomorphic_phi_function(9)[1], [1, 2, 4, 5, 7, 8])
        self.assertEqual(zygomorphic_phi_function(10)[1], [1, 3, 7, 9])
        self.assertEqual(zygomorphic_phi_function(6)[1], [1, 5])
        self.assertEqual(zygomorphic_phi_function(1)[1], [1])

    def test_invalid_input(self):
        # Test invalid input
        with self.assertRaises(ValueError):
            zygomorphic_phi_function(0)
        with self.assertRaises(ValueError):
            zygomorphic_phi_function(-5)

    def test_depict_coprimes(self):
        # Test the depict_coprimes function 
        try:
            depict_coprimes(10, [1, 3, 7, 9])
        except Exception as e:
            self.fail(f"depict_coprimes raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()