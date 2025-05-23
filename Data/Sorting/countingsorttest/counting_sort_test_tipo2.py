import unittest
from counting_sort import counting_sort

class TestCustomSort(unittest.TestCase):
    def test_custom_sort_ascending(self):
        data_list = [
            {"label": "Alice", "value": 88},
            {"label": "Bob", "value": 75},
            {"label": "Charlie", "value": 93},
            {"label": "David", "value": 85},
            {"label": "Eve", "value": 77}
        ]
        expected_list = [
            {"label": "Bob", "value": 75},
            {"label": "Eve", "value": 77},
            {"label": "David", "value": 85},
            {"label": "Alice", "value": 88},
            {"label": "Charlie", "value": 93}
        ]
        result_list = counting_sort(data_list, attr="value")
        self.assertEqual(result_list, expected_list)

    def test_custom_sort_descending(self):
        data_list = [
            {"label": "Alice", "value": 88},
            {"label": "Bob", "value": 75},
            {"label": "Charlie", "value": 93},
            {"label": "David", "value": 85},
            {"label": "Eve", "value": 77}
        ]
        expected_list = [
            {"label": "Charlie", "value": 93},
            {"label": "Alice", "value": 88},
            {"label": "David", "value": 85},
            {"label": "Eve", "value": 77},
            {"label": "Bob", "value": 75}
        ]
        result_list = counting_sort(data_list, attr="value", reverse=True)
        self.assertEqual(result_list, expected_list)

    def test_custom_sort_empty(self):
        data_list = []
        expected_list = []
        result_list = counting_sort(data_list, attr="value")
        self.assertEqual(result_list, expected_list)

    def test_custom_sort_invalid_attr(self):
        data_list = [{"label": "Alice", "value": 88}]
        with self.assertRaises(KeyError):
            counting_sort(data_list, attr="grade")

    def test_custom_sort_non_integer_attr(self):
        data_list = [{"label": "Alice", "value": "eighty-eight"}]
        with self.assertRaises(ValueError):
            counting_sort(data_list, attr="value")

    def test_custom_sort_negative_attr(self):
        data_list = [{"label": "Alice", "value": -10}]
        with self.assertRaises(ValueError):
            counting_sort(data_list, attr="value")

if __name__ == "__main__":
    unittest.main()