import unittest
from bubble_sort import bubble_sort

class TestCustomSorting(unittest.TestCase):
    def test_custom_sort_ascending(self):
        arr = [
            {"label": "Alpha", "value": 30},
            {"label": "Beta", "value": 25},
            {"label": "Gamma", "value": 35},
            {"label": "Delta", "value": 20}
        ]
        expected_result = [
            {"label": "Delta", "value": 20},
            {"label": "Beta", "value": 25},
            {"label": "Alpha", "value": 30},
            {"label": "Gamma", "value": 35}
        ]
        sorted_arr = bubble_sort(arr, key="value")
        self.assertEqual(sorted_arr, expected_result)

    def test_custom_sort_descending(self):
        arr = [
            {"label": "Alpha", "value": 30},
            {"label": "Beta", "value": 25},
            {"label": "Gamma", "value": 35},
            {"label": "Delta", "value": 20}
        ]
        expected_result = [
            {"label": "Gamma", "value": 35},
            {"label": "Alpha", "value": 30},
            {"label": "Beta", "value": 25},
            {"label": "Delta", "value": 20}
        ]
        sorted_arr = bubble_sort(arr, key="value", reverse=True)
        self.assertEqual(sorted_arr, expected_result)

    def test_custom_sort_empty(self):
        arr = []
        expected_result = []
        sorted_arr = bubble_sort(arr, key="value")
        self.assertEqual(sorted_arr, expected_result)

    def test_custom_sort_invalid_key(self):
        arr = [{"label": "Alpha", "value": 30}]
        with self.assertRaises(KeyError):
            bubble_sort(arr, key="height")

    def test_custom_sort_non_dict_input(self):
        arr = [1, 2, 3]
        with self.assertRaises(TypeError):
            bubble_sort(arr, key="value")

    def test_custom_sort_missing_key(self):
        arr = [{"label": "Alpha", "value": 30}, {"label": "Beta"}]
        with self.assertRaises(KeyError):
            bubble_sort(arr, key="value")

if __name__ == "__main__":
    unittest.main()