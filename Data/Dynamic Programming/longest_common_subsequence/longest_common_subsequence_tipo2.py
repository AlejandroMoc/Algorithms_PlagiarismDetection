from dataclasses import dataclass

import logging



@dataclass

class SeqInput:

    a: str

    b: str



def funky_sequence(input_data: SeqInput):

    try:

        a, b = input_data.a, input_data.b

        m, n = len(a), len(b)

        dp = [[0] * (n + 1) for _ in range(m + 1)]



        for i in range(1, m + 1):

            for j in range(1, n + 1):

                if a[i - 1] == b[j - 1]:

                    dp[i][j] = dp[i - 1][j - 1] + 1

                else:

                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])



        return dp[m][n]

    except Exception as e:

        logging.error(f"Error calculating Funky Seq: {e}")

        return None



# Test cases

def test_funky_sequence():

    test_cases = [

        (SeqInput("AGGTAB", "GXTXAYB"), 4),

        (SeqInput("ABCBDAB", "BDCAB"), 4),

        (SeqInput("", ""), 0),

        (SeqInput("ABC", "AC"), 2),

    ]

    

    for input_data, expected in test_cases:

        result = funky_sequence(input_data)

        assert result == expected, f"Expected {expected}, got {result}"



if __name__ == "__main__":

    test_funky_sequence()