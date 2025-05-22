import unittest
from merge_sort import merge_sort

class TestMergeSort(unittest.TestCase):
    def test_ascending_sort(self):
        data = [
            {"name": "Alice", "age": 28},
            {"name": "Bob", "age": 19},
            {"name": "Charlie", "age": 24},
            {"name": "David", "age": 22}
        ]
        expected = [
            {"name": "Bob", "age": 19},
            {"name": "David", "age": 22},
            {"name": "Charlie", "age": 24},
            {"name": "Alice", "age": 28}
        ]
        result = merge_sort(data, key="age")
        self.assertEqual(result, expected)

    def test_descending_sort(self):
        data = [
            {"name": "Alice", "age": 28},
            {"name": "Bob", "age": 19},
            {"name": "Charlie", "age": 24},
            {"name": "David", "age": 22}
        ]
        expected = [
            {"name": "Alice", "age": 28},
            {"name": "Charlie", "age": 24},
            {"name": "David", "age": 22},
            {"name": "Bob", "age": 19}
        ]
        result = merge_sort(data, key="age")
        result.reverse()
        self.assertEqual(result, expected)

    def test_empty_sort(self):
        data = []
        expected = []
        result = merge_sort(data, key="age")
        self.assertEqual(result, expected)

    def test_invalid_key_sort(self):
        data = [{"name": "Alice", "age": 28}]
        with self.assertRaises(KeyError):
            merge_sort(data, key="height")

    def test_non_dict_input_sort(self):
        data = ["alice", "bob", "charlie"]
        with self.assertRaises(TypeError):
            merge_sort(data, key="age")

    def test_missing_key_sort(self):
        data = [{"name": "Alice", "age": 28}, {"name": "Bob"}]
        with self.assertRaises(KeyError):
            merge_sort(data, key="age")

if __name__ == "__main__":
    unittest.main() 