import unittest
from funky_sort import funky_sort

class TestFunkySort(unittest.TestCase):
    def test_funky_sort_ascending(self):
        data = [
            {"name": "Alice", "age": 170},
            {"name": "Bob", "age": 45},
            {"name": "Charlie", "age": 75},
            {"name": "David", "age": 150}
        ]
        expected = [
            {"name": "Bob", "age": 45},
            {"name": "Charlie", "age": 75},
            {"name": "David", "age": 150},
            {"name": "Alice", "age": 170}
        ]
        result = funky_sort(data, key="age")
        self.assertEqual(result, expected)

    def test_funky_sort_empty(self):
        data = []
        expected = []
        result = funky_sort(data, key="age")
        self.assertEqual(result, expected)

    def test_funky_sort_invalid_key(self):
        data = [{"name": "Alice", "age": 30}]
        with self.assertRaises(KeyError):
            funky_sort(data, key="height")

    def test_funky_sort_non_integer_key(self):
        data = [{"name": "Alice", "age": "thirty"}]
        with self.assertRaises(ValueError):
            funky_sort(data, key="age")

    def test_funky_sort_negative_key(self):
        data = [{"name": "Alice", "age": -10}]
        with self.assertRaises(ValueError):
            funky_sort(data, key="age")

if __name__ == "__main__":
    unittest.main()