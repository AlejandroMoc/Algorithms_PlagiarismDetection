import unittest
from heap_sort import heap_sort

class TestHeapSortRenamed(unittest.TestCase):
    def test_heap_sort_ascending_renamed(self):
        dataset = [
            {"tag": "Alice", "years": 50},
            {"tag": "Bob", "years": 20},
            {"tag": "Charlie", "years": 40},
            {"tag": "David", "years": 30}
        ]
        projected = [
            {"tag": "Bob", "years": 20},
            {"tag": "David", "years": 30},
            {"tag": "Charlie", "years": 40},
            {"tag": "Alice", "years": 50}
        ]
        resultset = heap_sort(dataset, key="years")
        self.assertEqual(resultset, projected)

    def test_heap_sort_descending_renamed(self):
        dataset = [
            {"tag": "Alice", "years": 50},
            {"tag": "Bob", "years": 20},
            {"tag": "Charlie", "years": 40},
            {"tag": "David", "years": 30}
        ]
        projected = [
            {"tag": "Alice", "years": 50},
            {"tag": "Charlie", "years": 40},
            {"tag": "David", "years": 30},
            {"tag": "Bob", "years": 20}
        ]
        resultset = heap_sort(dataset, key="years", reverse=True)
        self.assertEqual(resultset, projected)

    def test_heap_sort_empty_renamed(self):
        dataset = []
        projected = []
        resultset = heap_sort(dataset, key="years")
        self.assertEqual(resultset, projected)

    def test_heap_sort_invalid_key_renamed(self):
        dataset = [{"tag": "Alice", "years": 50}]
        with self.assertRaises(KeyError):
            heap_sort(dataset, key="length")

    def test_heap_sort_non_dict_input_renamed(self):
        dataset = [100, 50, 200]
        with self.assertRaises(TypeError):
            heap_sort(dataset, key="years")

    def test_heap_sort_missing_key_renamed(self):
        dataset = [{"tag": "Alice", "years": 50}, {"tag": "Bob"}]
        with self.assertRaises(KeyError):
            heap_sort(dataset, key="years")

if __name__ == "__main__":
    unittest.main()