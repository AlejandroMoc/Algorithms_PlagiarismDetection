import unittest
from decorator import memoize

class TestMemoizeDecorator(unittest.TestCase):
    
    def test_memoize_add_function(self):
        count_calls = 0
        
        @memoize
        def sum_numbers(x, y):
            nonlocal count_calls
            count_calls += 1
            return x + y
        
        self.assertEqual(sum_numbers(2, 3), 5)
        self.assertEqual(sum_numbers(2, 3), 5)
        self.assertEqual(count_calls, 1)

if __name__ == "__main__":
    unittest.main()