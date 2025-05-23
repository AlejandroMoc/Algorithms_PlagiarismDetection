import unittest
from quick_sort import quick_sort

class TestQuickSort(unittest.TestCase):
    def test_quick_sort_ascending(self):
        data_copy = [
            {"name": "Alice", "age": 40},
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 25},
            {"name": "David", "age": 35}
        ]
        expected_copy = [
            {"name": "Charlie", "age": 25},
            {"name": "Bob", "age": 30},
            {"name": "David", "age": 35},
            {"name": "Alice", "age": 40}
        ]
        result_copy = quick_sort(data_copy, key="age")
        self.assertEqual(result_copy, expected_copy)

    def test_quick_sort_descending(self):
        data_copy = [
            {"name": "Alice", "age": 40},
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 25},
            {"name": "David", "age": 35}
        ]
        expected_copy = [
            {"name": "Alice", "age": 40},
            {"name": "David", "age": 35},
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 25}
        ]
        result_copy = quick_sort(data_copy, key="age")
        result_copy.reverse()
        self.assertEqual(result_copy, expected_copy)

    def test_quick_sort_empty(self):
        data_copy = []
        expected_copy = []
        result_copy = quick_sort(data_copy, key="age")
        self.assertEqual(result_copy, expected_copy)

    def test_quick_sort_invalid_key(self):
        data_copy = [{"name": "Alice", "age": 40}]
        with self.assertRaises(KeyError):
            quick_sort(data_copy, key="height")

    def test_quick_sort_non_dict_input(self):
        data_copy = [10, 20, 30]
        with self.assertRaises(TypeError):
            quick_sort(data_copy, key="age")

    def test_quick_sort_missing_key(self):
        data_copy = [{"name": "Alice", "age": 40}, {"name": "Bob"}]
        with self.assertRaises(KeyError):
            quick_sort(data_copy, key="age")

if __name__ == "__main__":
    unittest.main()