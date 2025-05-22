import unittest
from dif_dist import dif_dist, DifDistInput
from lcs_algo import lcs_algo, LCSInput
from coin_combo import coin_combo, CoinComboInput
from max_sum_sub import max_sum_sub, MaxSumSubInput
from Longest_path_in_DAG import find_longest_path  # Assuming the function is correctly named
from unique_ways import unique_ways

class TestDynamicProgramming(unittest.TestCase):

    def test_dif_dist(self):
        test_cases = [
            (DifDistInput("horse", "ros"), 3),
            (DifDistInput("intention", "execution"), 5),
            (DifDistInput("", ""), 0),
            (DifDistInput("a", "b"), 1),
            (DifDistInput("abc", "yabd"), 2),
        ]
        
        for input_data, expected in test_cases:
            result = dif_dist(input_data)
            self.assertEqual(result, expected)

    def test_lcs_algo(self):
        test_cases = [
            (LCSInput("AGGTAB", "GXTXAYB"), 4),
            (LCSInput("ABCBDAB", "BDCAB"), 4),
            (LCSInput("", ""), 0),
            (LCSInput("ABC", "AC"), 2),
        ]
        
        for input_data, expected in test_cases:
            result = lcs_algo(input_data)
            self.assertEqual(result, expected)

    def test_coin_combo(self):
        test_cases = [
            (CoinComboInput([1, 2, 5], 11), (3, [5, 5, 1])),
            (CoinComboInput([2], 3), -1),
            (CoinComboInput([1], 0), (0, [])),
        ]
        
        for input_data, expected in test_cases:
            result = coin_combo(input_data)
            self.assertEqual(result, expected)

    def test_max_sum_sub(self):
        test_cases = [
            (MaxSumSubInput([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6),
            (MaxSumSubInput([1]), 1),
            (MaxSumSubInput([0, -1, 2, -3, 4]), 4),
        ]
        
        for input_data, expected in test_cases:
            result = max_sum_sub(input_data)
            self.assertEqual(result, expected)

    def test_unique_ways(self):
        test_cases = [
            (3, 7, 28),
            (3, 2, 3),
            (7, 3, 28)
        ]
        
        for m, n, expected in test_cases:
            result = unique_ways(m, n)
            self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()