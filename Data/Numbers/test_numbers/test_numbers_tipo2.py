import unittest
from num_format_2_decimals import num_format_2_decimals
from num_formatting import num_format
from zero_pad import zero_pad
from perc_format import perc_format
from factorize_primes import factorize_primes

class TestNumFormatting(unittest.TestCase):

    def test_num_format_2_decimals(self):
        test_cases = [
            (123.456, '123.46'),
            (123, '123.00'),
            (-123.456, '-123.46')
        ]
        for num, expected in test_cases:
            self.assertEqual(num_format_2_decimals(num), expected)

    def test_num_format(self):
        test_cases = [
            (1234567, '1,234,567'),
            (12345.6789, '12 345,6789', {'thousands_sep': ' ', 'dec_sep': ','})
        ]
        for num, expected, *args in test_cases:
            self.assertEqual(num_format(num, **(args[0] if args else {})), expected)

    def test_zero_pad(self):
        test_cases = [
            (123, 6, '000123'),
            (1, 3, '001')
        ]
        for num, width, expected in test_cases:
            self.assertEqual(zero_pad(num, width), expected)

    def test_perc_format(self):
        test_cases = [
            (50, 200, '25.00%'),
            (1, 3, 3, '33.333%')
        ]
        for num, total, *args in test_cases:
            self.assertEqual(perc_format(num, total, *(args if args else [])), expected)

    def test_factorize_primes(self):
        test_cases = [
            (28, [2, 2, 7]),
            (1, []),
            (13, [13])
        ]
        for num, expected in test_cases:
            self.assertEqual(factorize_primes(num), expected)

if __name__ == '__main__':
    unittest.main()