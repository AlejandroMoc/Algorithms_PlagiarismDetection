import unittest
from sorting_bucket import sorting_bucket

class TestSortingBucket(unittest.TestCase):
    def test_sorting_bucket_ascending(self):
        data = [
            {"title": "Alpha", "value": 88.5},
            {"title": "Beta", "value": 75.0},
            {"title": "Gamma", "value": 93.2},
            {"title": "Delta", "value": 85.3},
            {"title": "Epsilon", "value": 77.8}
        ]
        expected = [
            {"title": "Beta", "value": 75.0},
            {"title": "Epsilon", "value": 77.8},
            {"title": "Delta", "value": 85.3},
            {"title": "Alpha", "value": 88.5},
            {"title": "Gamma", "value": 93.2}
        ]
        result = sorting_bucket(data, key="value")
        self.assertEqual(result, expected)
        
    def test_sorting_bucket_empty(self):
        data = []
        expected = []
        result = sorting_bucket(data, key="value")
        self.assertEqual(result, expected)

    def test_sorting_bucket_invalid_key(self):
        data = [{"title": "Alpha", "value": 88.5}]
        with self.assertRaises(KeyError):
            sorting_bucket(data, key="grade")

    def test_sorting_bucket_non_numeric_key(self):
        data = [{"title": "Alpha", "value": "eighty"}]
        with self.assertRaises(ValueError):
            sorting_bucket(data, key="value")

if __name__ == "__main__":
    unittest.main()