import unittest
from rapid_sort import rapid_sort

class TestRapidSort(unittest.TestCase):
    def test_rapid_sort_ascending(self):
        information = [
            {"tag": "Alice", "quality": 40},
            {"tag": "Bob", "quality": 30},
            {"tag": "Charlie", "quality": 25},
            {"tag": "David", "quality": 35}
        ]
        projected = [
            {"tag": "Charlie", "quality": 25},
            {"tag": "Bob", "quality": 30},
            {"tag": "David", "quality": 35},
            {"tag": "Alice", "quality": 40}
        ]
        result = rapid_sort(information, key="quality")
        self.assertEqual(result, projected)

    def test_rapid_sort_descending(self):
        information = [
            {"tag": "Alice", "quality": 40},
            {"tag": "Bob", "quality": 30},
            {"tag": "Charlie", "quality": 25},
            {"tag": "David", "quality": 35}
        ]
        # Rapid Sort doesn't support reverse, so we'll sort ascending and then reverse
        projected = [
            {"tag": "Alice", "quality": 40},
            {"tag": "David", "quality": 35},
            {"tag": "Bob", "quality": 30},
            {"tag": "Charlie", "quality": 25}
        ]
        result = rapid_sort(information, key="quality")
        result.reverse()
        self.assertEqual(result, projected)

    def test_rapid_sort_empty(self):
        information = []
        projected = []
        result = rapid_sort(information, key="quality")
        self.assertEqual(result, projected)

    def test_rapid_sort_invalid_key(self):
        information = [{"tag": "Alice", "quality": 40}]
        with self.assertRaises(KeyError):
            rapid_sort(information, key="stature")

    def test_rapid_sort_non_dict_input(self):
        information = [10, 20, 30]
        with self.assertRaises(TypeError):
            rapid_sort(information, key="quality")

    def test_rapid_sort_missing_key(self):
        information = [{"tag": "Alice", "quality": 40}, {"tag": "Bob"}]
        with self.assertRaises(KeyError):
            rapid_sort(information, key="quality")

if __name__ == "__main__":
    unittest.main()