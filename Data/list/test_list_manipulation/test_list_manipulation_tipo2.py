import unittest
from lstutils import LstUtil

class TestLstUtil(unittest.TestCase):

    def test_deduplicate_elements(self):
        self.assertEqual(LstUtil.deduplicate_elements([1, 2, 2, 3, 4, 4]), [1, 2, 3, 4])
        self.assertEqual(LstUtil.deduplicate_elements([1, 1, 1]), [1])

    def test_flatten_nested_lst(self):
        self.assertEqual(LstUtil.flatten_nested_lst([[1, 2], [3, 4], [5]]), [1, 2, 3, 4, 5])
        self.assertEqual(LstUtil.flatten_nested_lst([]), [])

    def test_lst_intersection(self):
        self.assertEqual(LstUtil.lst_intersection([1, 2, 3], [2, 3, 4]), [2, 3])
        self.assertEqual(LstUtil.lst_intersection([1, 2], [3, 4]), [])

    def test_shuffle_randomly(self):
        shuffled = LstUtil.shuffle_randomly([1, 2, 3, 4])
        self.assertEqual(sorted(shuffled), [1, 2, 3, 4])  # Check if all elements are present

    def test_sort_by_occurrence(self):
        self.assertEqual(LstUtil.sort_by_occurrence([4, 5, 6, 4, 5, 4]), [4, 4, 4, 5, 5, 6])
        self.assertEqual(LstUtil.sort_by_occurrence([1, 2, 2, 3, 3, 3]), [3, 3, 3, 2, 2, 1])

    def test_chunk_lst(self):
        self.assertEqual(LstUtil.chunk_lst([1, 2, 3, 4, 5, 6], 2), [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(LstUtil.chunk_lst([1, 2, 3, 4, 5], 3), [[1, 2, 3], [4, 5]])

    def test_most_occurred_element(self):
        self.assertEqual(LstUtil.most_occurred_element([1, 2, 2, 3, 3, 3]), 3)
        self.assertIsNone(LstUtil.most_occurred_element([]))

    def test_rotate_lst(self):
        self.assertEqual(LstUtil.rotate_lst([1, 2, 3, 4, 5], 2), [4, 5, 1, 2, 3])
        self.assertEqual(LstUtil.rotate_lst([1, 2, 3, 4, 5], 5), [1, 2, 3, 4, 5])

    def test_unique_elems(self):
        self.assertEqual(LstUtil.unique_elems([1, 2, 2, 3, 3, 4]), [1, 4])
        self.assertEqual(LstUtil.unique_elems([1, 1, 1, 2]), [2])

    def test_sum_pairs(self):
        self.assertEqual(LstUtil.sum_pairs([1, 2, 3, 4, 5], 6), [(2, 4), (1, 5)])
        self.assertEqual(LstUtil.sum_pairs([1, 2, 3], 7), [])

if __name__ == '__main__':
    unittest.main()