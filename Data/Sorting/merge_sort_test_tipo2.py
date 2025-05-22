import unittest
from funky_sort import funky_sorter

class TestFunkySorter(unittest.TestCase):
    def test_funky_sorter_ascending(self):
        funky_data = [
            {"id": "Alpha", "num": 28},
            {"id": "Beta", "num": 19},
            {"id": "Gamma", "num": 24},
            {"id": "Delta", "num": 22}
        ]
        expected_data = [
            {"id": "Beta", "num": 19},
            {"id": "Delta", "num": 22},
            {"id": "Gamma", "num": 24},
            {"id": "Alpha", "num": 28}
        ]
        result_data = funky_sorter(funky_data, key="num")
        self.assertEqual(result_data, expected_data)

    def test_funky_sorter_descending(self):
        funky_data = [
            {"id": "Alpha", "num": 28},
            {"id": "Beta", "num": 19},
            {"id": "Gamma", "num": 24},
            {"id": "Delta", "num": 22}
        ]
        expected_data = [
            {"id": "Alpha", "num": 28},
            {"id": "Gamma", "num": 24},
            {"id": "Delta", "num": 22},
            {"id": "Beta", "num": 19}
        ]
        result_data = funky_sorter(funky_data, key="num")
        result_data.reverse()
        self.assertEqual(result_data, expected_data)

    def test_funky_sorter_empty(self):
        funky_data = []
        expected_data = []
        result_data = funky_sorter(funky_data, key="num")
        self.assertEqual(result_data, expected_data)

    def test_funky_sorter_invalid_key(self):
        funky_data = [{"id": "Alpha", "num": 28}]
        with self.assertRaises(KeyError):
            funky_sorter(funky_data, key="size")

    def test_funky_sorter_non_dict_input(self):
        funky_data = ["alpha", "beta", "gamma"]
        with self.assertRaises(TypeError):
            funky_sorter(funky_data, key="num")

    def test_funky_sorter_missing_key(self):
        funky_data = [{"id": "Alpha", "num": 28}, {"id": "Beta"}]
        with self.assertRaises(KeyError):
            funky_sorter(funky_data, key="num")

if __name__ == "__main__":
    unittest.main()