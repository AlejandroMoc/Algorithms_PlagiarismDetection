import unittest
from cocktail_shaker_sort import cocktail_shaker_sort

class TestCocktailShakerSort(unittest.TestCase):
    def test_cocktail_shaker_sort_ascending(self):
        data_list = [5, 3, 8, 4, 2]
        expected_result = [2, 3, 4, 5, 8]
        result_data = cocktail_shaker_sort(data_list)
        self.assertEqual(result_data, expected_result)

    def test_cocktail_shaker_sort_descending(self):
        data_list = [5, 3, 8, 4, 2]
        expected_result = [8, 5, 4, 3, 2]
        sorted_data = cocktail_shaker_sort(data_list)
        sorted_data.reverse()
        self.assertEqual(sorted_data, expected_result)

    def test_cocktail_shaker_sort_empty(self):
        data_list = []
        expected_result = []
        result_data = cocktail_shaker_sort(data_list)
        self.assertEqual(result_data, expected_result)

    def test_cocktail_shaker_sort_single_element(self):
        data_list = [1]
        expected_result = [1]
        result_data = cocktail_shaker_sort(data_list)
        self.assertEqual(result_data, expected_result)

    def test_cocktail_shaker_sort_already_sorted(self):
        data_list = [1, 2, 3, 4, 5]
        expected_result = [1, 2, 3, 4, 5]
        result_data = cocktail_shaker_sort(data_list)
        self.assertEqual(result_data, expected_result)

    def test_cocktail_shaker_sort_reverse_sorted(self):
        data_list = [5, 4, 3, 2, 1]
        expected_result = [1, 2, 3, 4, 5]
        result_data = cocktail_shaker_sort(data_list)
        self.assertEqual(result_data, expected_result)

    def test_cocktail_shaker_sort_with_duplicates(self):
        data_list = [4, 2, 5, 2, 3, 4]
        expected_result = [2, 2, 3, 4, 4, 5]
        result_data = cocktail_shaker_sort(data_list)
        self.assertEqual(result_data, expected_result)

if __name__ == "__main__":
    unittest.main()