from dataclasses import dataclass

import logging



@dataclass

class StrEditInput:

    txt1: str

    txt2: str



def str_edit(input_info: StrEditInput):

    try:

        txt1, txt2 = input_info.txt1, input_info.txt2

        len1, len2 = len(txt1), len(txt2)

        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]



        for i in range(len1 + 1):

            for j in range(len2 + 1):

                if i == 0:

                    matrix[i][j] = j

                elif j == 0:

                    matrix[i][j] = i

                elif txt1[i - 1] == txt2[j - 1]:

                    matrix[i][j] = matrix[i - 1][j - 1]

                else:

                    matrix[i][j] = 1 + min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1])



        return matrix[len1][len2]

    except Exception as ex:

        logging.error(f"Error calculating string edit distance: {ex}")

        return None



# Test cases

def test_str_edit():

    test_cases = [

        (StrEditInput("horse", "ros"), 3),

        (StrEditInput("intention", "execution"), 5),

        (StrEditInput("", ""), 0),

        (StrEditInput("a", "b"), 1),

        (StrEditInput("abc", "yabd"), 2),

    ]

    

    for input_info, expected_val in test_cases:

        result = str_edit(input_info)

        assert result == expected_val, f"Expected {expected_val}, got {result}"



if __name__ == "__main__":

    test_str_edit()