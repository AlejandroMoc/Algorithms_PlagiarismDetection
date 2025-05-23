import unittest
from misterio import SecretSolver, SecretItem

class TestSecretSolver(unittest.TestCase):
    
    def test_secret_standard_case(self):
        items = (
            SecretItem(value=60, weight=10),
            SecretItem(value=100, weight=20),
            SecretItem(value=120, weight=30),
        )
        self.assertEqual(SecretSolver.secret_method(50, items, 3), 220)
    
    def test_secret_no_items(self):
        items = ()
        self.assertEqual(SecretSolver.secret_method(50, items, 0), 0)
    
    def test_secret_zero_capacity(self):
        items = (
            SecretItem(value=60, weight=10),
        )
        self.assertEqual(SecretSolver.secret_method(0, items, 1), 0)
    
    def test_secret_item_exceeds_capacity(self):
        items = (
            SecretItem(value=100, weight=60),
        )
        self.assertEqual(SecretSolver.secret_method(50, items, 1), 0)

if __name__ == "__main__":
    unittest.main()