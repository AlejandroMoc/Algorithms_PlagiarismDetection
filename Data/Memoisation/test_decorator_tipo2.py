import unittest
from decorator import memoize

class TestMemoizeDecorator(unittest.TestCase):
    
    def test_memoize_add_function(self):
        counter = 0
        
        @memoize
        def combine_strings(s1, s2):
            nonlocal counter
            counter += 1
            return s1 + s2
        
        self.assertEqual(combine_strings("hello", "world"), "helloworld")
        self.assertEqual(combine_strings("hello", "world"), "helloworld")
        self.assertEqual(counter, 1)

if __name__ == "__main__":
    unittest.main()