from dataclasses import dataclass
import logging

@dataclass
class LongestCommonSubsequenceInput:
    str1: str
    str2: str

def find_longest_common_subsequence(input_data: LongestCommonSubsequenceInput):
    try:
        str1, str2 = input_data.str1, input_data.str2
        len_str1, len_str2 = len(str1), len(str2)
        dp = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]

        for i in range(1, len_str1 + 1):
            for j in range(1, len_str2 + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[len_str1][len_str2]
    except Exception as e:
        logging.error(f"Error finding LCS: {e}")
        return None

# Test cases
def test_find_longest_common_subsequence():
    test_cases = [
        (LongestCommonSubsequenceInput("AGGTAB", "GXTXAYB"), 4),
        (LongestCommonSubsequenceInput("ABCBDAB", "BDCAB"), 4),
        (LongestCommonSubsequenceInput("", ""), 0),
        (LongestCommonSubsequenceInput("ABC", "AC"), 2),
    ]

    for input_data, expected in test_cases:
        result = find_longest_common_subsequence(input_data)
        assert result == expected, f"Expected {expected}, got {result}"

if __name__ == "__main__":
    test_find_longest_common_subsequence()