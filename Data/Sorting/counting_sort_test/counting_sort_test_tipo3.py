import unittest
from counting_sort import counting_sort

class TestCountingSort(unittest.TestCase):
    def test_counting_sort_ascending(self):
        data_test = [
            {"name": "Alice", "score": 88},
            {"name": "Bob", "score": 75},
            {"name": "Charlie", "score": 93},
            {"name": "David", "score": 85},
            {"name": "Eve", "score": 77}
        ]
        expected_result = [
            {"name": "Bob", "score": 75},
            {"name": "Eve", "score": 77},
            {"name": "David", "score": 85},
            {"name": "Alice", "score": 88},
            {"name": "Charlie", "score": 93}
        ]
        actual_result = counting_sort(data_test, key="score")
        self.assertEqual(actual_result, expected_result)

    def test_counting_sort_descending(self):
        data_test = [
            {"name": "Alice", "score": 88},
            {"name": "Bob", "score": 75},
            {"name": "Charlie", "score": 93},
            {"name": "David", "score": 85},
            {"name": "Eve", "score": 77}
        ]
        expected_result = [
            {"name": "Charlie", "score": 93},
            {"name": "Alice", "score": 88},
            {"name": "David", "score": 85},
            {"name": "Eve", "score": 77},
            {"name": "Bob", "score": 75}
        ]
        actual_result = counting_sort(data_test, key="score", reverse=True)
        self.assertEqual(actual_result, expected_result)

    def test_counting_sort_empty(self):
        data_test = []
        expected_result = []
        actual_result = counting_sort(data_test, key="score")
        self.assertEqual(actual_result, expected_result)

    def test_counting_sort_invalid_key(self):
        data_test = [{"name": "Alice", "score": 88}]
        with self.assertRaises(KeyError):
            counting_sort(data_test, key="grade")

    def test_counting_sort_non_integer_key(self):
        data_test = [{"name": "Alice", "score": "eighty-eight"}]
        with self.assertRaises(ValueError):
            counting_sort(data_test, key="score")

    def test_counting_sort_negative_key(self):
        data_test = [{"name": "Alice", "score": -10}]
        with self.assertRaises(ValueError):
            counting_sort(data_test, key="score")

if __name__ == "__main__":
    unittest.main()