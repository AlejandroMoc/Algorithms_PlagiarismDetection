import unittest
from insort import insort

class TestInsort(unittest.TestCase):
    def test_insort_ascending(self):
        data = [
            {"label": "Alice", "num": 34},
            {"label": "Bob", "num": 23},
            {"label": "Charlie", "num": 45},
            {"label": "David", "num": 29}
        ]
        expected = [
            {"label": "Bob", "num": 23},
            {"label": "David", "num": 29},
            {"label": "Alice", "num": 34},
            {"label": "Charlie", "num": 45}
        ]
        result = insort(data, key="num")
        self.assertEqual(result, expected)

    def test_insort_descending(self):
        data = [
            {"label": "Alice", "num": 34},
            {"label": "Bob", "num": 23},
            {"label": "Charlie", "num": 45},
            {"label": "David", "num": 29}
        ]
        expected = [
            {"label": "Charlie", "num": 45},
            {"label": "Alice", "num": 34},
            {"label": "David", "num": 29},
            {"label": "Bob", "num": 23}
        ]
        result = insort(data, key="num", rev=True)
        self.assertEqual(result, expected)

    def test_insort_empty(self):
        data = []
        expected = []
        result = insort(data, key="num")
        self.assertEqual(result, expected)

    def test_insort_invalid_key(self):
        data = [{"label": "Alice", "num": 34}]
        with self.assertRaises(KeyError):
            insort(data, key="height")

    def test_insort_non_dict_input(self):
        data = [10, 20, 30]
        with self.assertRaises(TypeError):
            insort(data, key="num")

    def test_insort_missing_key(self):
        data = [{"label": "Alice", "num": 34}, {"label": "Bob"}]
        with self.assertRaises(KeyError):
            insort(data, key="num")

if __name__ == "__main__":
    unittest.main()