import unittest
from bubble_sort import bubble_sort

class TestBubbleSort(unittest.TestCase):
    def test_bubble_sort_ascending(self):
        sample_data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
            {"name": "David", "age": 20}
        ]
        expected_result = [
            {"name": "David", "age": 20},
            {"name": "Bob", "age": 25},
            {"name": "Alice", "age": 30},
            {"name": "Charlie", "age": 35}
        ]
        final_result = bubble_sort(sample_data, key="age")
        self.assertEqual(final_result, expected_result)

    def test_bubble_sort_descending(self):
        sample_data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
            {"name": "David", "age": 20}
        ]
        expected_result = [
            {"name": "Charlie", "age": 35},
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "David", "age": 20}
        ]
        final_result = bubble_sort(sample_data, key="age", reverse=True)
        self.assertEqual(final_result, expected_result)

    def test_bubble_sort_empty(self):
        sample_data = []
        expected_result = []
        final_result = bubble_sort(sample_data, key="age")
        self.assertEqual(final_result, expected_result)

    def test_bubble_sort_invalid_key(self):
        sample_data = [{"name": "Alice", "age": 30}]
        with self.assertRaises(KeyError):
            bubble_sort(sample_data, key="height")

    def test_bubble_sort_non_dict_input(self):
        sample_data = [1, 2, 3]
        with self.assertRaises(TypeError):
            bubble_sort(sample_data, key="age")

    def test_bubble_sort_missing_key(self):
        sample_data = [{"name": "Alice", "age": 30}, {"name": "Bob"}]
        with self.assertRaises(KeyError):
            bubble_sort(sample_data, key="age")

if __name__ == "__main__":
    unittest.main()