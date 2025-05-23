import unittest
from shuffling_sort import shuffling_sort

class TestShufflingSort(unittest.TestCase):
    def test_shuffling_sort_ascending(self):
        data_list = [5, 3, 8, 4, 2]
        expected_list = [2, 3, 4, 5, 8]
        result_list = shuffling_sort(data_list)
        self.assertEqual(result_list, expected_list)

    def test_shuffling_sort_descending(self):
        data_list = [5, 3, 8, 4, 2]
        expected_list = [8, 5, 4, 3, 2]
        sorted_data_list = shuffling_sort(data_list)
        sorted_data_list.reverse()
        self.assertEqual(sorted_data_list, expected_list)

    def test_shuffling_sort_empty(self):
        data_list = []
        expected_list = []
        result_list = shuffling_sort(data_list)
        self.assertEqual(result_list, expected_list)

    def test_shuffling_sort_single_element(self):
        data_list = [1]
        expected_list = [1]
        result_list = shuffling_sort(data_list)
        self.assertEqual(result_list, expected_list)

    def test_shuffling_sort_already_sorted(self):
        data_list = [1, 2, 3, 4, 5]
        expected_list = [1, 2, 3, 4, 5]
        result_list = shuffling_sort(data_list)
        self.assertEqual(result_list, expected_list)

    def test_shuffling_sort_reverse_sorted(self):
        data_list = [5, 4, 3, 2, 1]
        expected_list = [1, 2, 3, 4, 5]
        result_list = shuffling_sort(data_list)
        self.assertEqual(result_list, expected_list)

    def test_shuffling_sort_with_duplicates(self):
        data_list = [4, 2, 5, 2, 3, 4]
        expected_list = [2, 2, 3, 4, 4, 5]
        result_list = shuffling_sort(data_list)
        self.assertEqual(result_list, expected_list)

if __name__ == "__main__":
    unittest.main()