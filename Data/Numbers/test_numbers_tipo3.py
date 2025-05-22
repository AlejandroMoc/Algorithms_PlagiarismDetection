import unittest
from format_to_2_decimal import format_to_2_decimal as f2d
from number_formatting import format_number as fn
from pad_with_zeros import pad_with_zeros as pwz
from percentage_format import percentage_format as pf
from prime_factorization import prime_factorization as pfact

class TestNumberFormatting(unittest.TestCase):

    def test_format_to_2_decimal(self):
        test_cases = [
            (123.456, '123.46'),
            (123, '123.00'),
            (-123.456, '-123.46')
        ]
        for num, expected in test_cases:
            self.assertEqual(f2d(num), expected)

    def test_format_number(self):
        test_cases = [
            (1234567, '1,234,567'),
            (12345.6789, '12 345,6789', {'thousands_sep': ' ', 'decimal_sep': ','})
        ]
        for num, expected, *args in test_cases:
            self.assertEqual(fn(num, **(args[0] if args else {})), expected)

    def test_pad_with_zeros(self):
        test_cases = [
            (123, 6, '000123'),
            (1, 3, '001')
        ]
        for num, width, expected in test_cases:
            self.assertEqual(pwz(num, width), expected)

    def test_percentage_format(self):
        test_cases = [
            (50, 200, '25.00%'),
            (1, 3, 3, '33.333%')
        ]
        for num, total, *args in test_cases:
            self.assertEqual(pf(num, total, *(args if args else [])), expected)

    def test_prime_factorization(self):
        test_cases = [
            (28, [2, 2, 7]),
            (1, []),
            (13, [13])
        ]
        for num, expected in test_cases:
            self.assertEqual(pfact(num), expected)

if __name__ == '__main__':
    unittest.main()