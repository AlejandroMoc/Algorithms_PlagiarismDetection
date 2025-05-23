import unittest
from heap_sort import heap_sort

class TestHeapSort(unittest.TestCase):
    def test_heap_sort_ascending(self):
        input_data = [
            {"name": "Alice", "age": 50},
            {"name": "Bob", "age": 20},
            {"name": "Charlie", "age": 40},
            {"name": "David", "age": 30}
        ]
        expected_output = [
            {"name": "Bob", "age": 20},
            {"name": "David", "age": 30},
            {"name": "Charlie", "age": 40},
            {"name": "Alice", "age": 50}
        ]
        result = heap_sort(input_data, key="age")
        self.assertEqual(result, expected_output)

    def test_heap_sort_descending(self):
        input_data = [
            {"name": "Alice", "age": 50},
            {"name": "Bob", "age": 20},
            {"name": "Charlie", "age": 40},
            {"name": "David", "age": 30}
        ]
        expected_output = [
            {"name": "Alice", "age": 50},
            {"name": "Charlie", "age": 40},
            {"name": "David", "age": 30},
            {"name": "Bob", "age": 20}
        ]
        result = heap_sort(input_data, key="age", reverse=True)
        self.assertEqual(result, expected_output)

    def test_heap_sort_empty(self):
        input_data = []
        expected_output = []
        result = heap_sort(input_data, key="age")
        self.assertEqual(result, expected_output)

    def test_heap_sort_invalid_key(self):
        input_data = [{"name": "Alice", "age": 50}]
        with self.assertRaises(KeyError):
            heap_sort(input_data, key="height")

    def test_heap_sort_non_dict_input(self):
        input_data = [100, 50, 200]
        with self.assertRaises(TypeError):
            heap_sort(input_data, key="age")

    def test_heap_sort_missing_key(self):
        input_data = [{"name": "Alice", "age": 50}, {"name": "Bob"}]
        with self.assertRaises(KeyError):
            heap_sort(input_data, key="age")

if __name__ == "__main__":
    unittest.main()