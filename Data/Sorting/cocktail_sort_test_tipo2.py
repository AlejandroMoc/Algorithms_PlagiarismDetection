import unittest
from funky_sorting import funky_sorting_algorithm

class TestFunkySorting(unittest.TestCase):
    def test_funky_sorting_ascending(self):
        data_list = [
            {"label": "Alice", "points": 45},
            {"label": "Bob", "points": 22},
            {"label": "Charlie", "points": 33},
            {"label": "David", "points": 27}
        ]
        expected_list = [
            {"label": "Bob", "points": 22},
            {"label": "David", "points": 27},
            {"label": "Charlie", "points": 33},
            {"label": "Alice", "points": 45}
        ]
        result_list = funky_sorting_algorithm(data_list, key="points")
        self.assertEqual(result_list, expected_list)

    def test_funky_sorting_descending(self):
        data_list = [
            {"label": "Alice", "points": 45},
            {"label": "Bob", "points": 22},
            {"label": "Charlie", "points": 33},
            {"label": "David", "points": 27}
        ]
        expected_list = [
            {"label": "Alice", "points": 45},
            {"label": "Charlie", "points": 33},
            {"label": "David", "points": 27},
            {"label": "Bob", "points": 22}
        ]
        result_list = funky_sorting_algorithm(data_list, key="points", reverse=True)
        self.assertEqual(result_list, expected_list)

    def test_funky_sorting_empty(self):
        data_list = []
        expected_list = []
        result_list = funky_sorting_algorithm(data_list, key="points")
        self.assertEqual(result_list, expected_list)

    def test_funky_sorting_invalid_key(self):
        data_list = [{"label": "Alice", "points": 45}]
        with self.assertRaises(KeyError):
            funky_sorting_algorithm(data_list, key="height")

    def test_funky_sorting_non_dict_input(self):
        data_list = ["Alice", "Bob", "Charlie"]
        with self.assertRaises(TypeError):
            funky_sorting_algorithm(data_list, key="points")

    def test_funky_sorting_missing_key(self):
        data_list = [{"label": "Alice", "points": 45}, {"label": "Bob"}]
        with self.assertRaises(KeyError):
            funky_sorting_algorithm(data_list, key="points")

if __name__ == "__main__":
    unittest.main()