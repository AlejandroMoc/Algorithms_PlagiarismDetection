import unittest
from bit_count_utils import calculate_set_bits
from unique_finder import find_unique_element
from number_swapper import swap_values
from power_checker import check_power_of_two
from bit_flipper import flip_bit_at_position
from clear_bit_util import clear_bits_up_to_position
from rightmost_one_isolator import isolate_rightmost_one_bit

class TestAdvancedBitManipulation(unittest.TestCase):

    def test_calculate_set_bits(self):
        self.assertEqual(calculate_set_bits(5), 2)  # 5 in binary is 101
        self.assertEqual(calculate_set_bits(15), 4)  # 15 in binary is 1111

    def test_find_unique_element(self):
        self.assertEqual(find_unique_element([1, 2, 2, 3, 1]), 3)
        self.assertEqual(find_unique_element([4, 3, 4, 5, 3]), 5)

    def test_swap_values(self):
        a, b = swap_values(5, 7)
        self.assertEqual(a, 7)
        self.assertEqual(b, 5)

    def test_check_power_of_two(self):
        self.assertTrue(check_power_of_two(8))  # 8 is 2^3
        self.assertFalse(check_power_of_two(10))

    def test_flip_bit_at_position(self):
        self.assertEqual(flip_bit_at_position(5, 0), 4)  # 5 is 101, flipping LSB gives 100 (4)
        self.assertEqual(flip_bit_at_position(5, 2), 1)  # 5 is 101, flipping bit at position 2 gives 001 (1)

    def test_clear_bits_up_to_position(self):
        self.assertEqual(clear_bits_up_to_position(15, 2), 12)  # Clears LSBs up to pos 2 in 1111, gives 1100 (12)
        self.assertEqual(clear_bits_up_to_position(29, 3), 24)  # 29 is 11101, clearing LSBs up to pos 3 gives 11000 (24)

    def test_isolate_rightmost_one_bit(self):
        self.assertEqual(isolate_rightmost_one_bit(12), 4)  # 12 is 1100, isolates rightmost 1 gives 0100 (4)
        self.assertEqual(isolate_rightmost_one_bit(18), 2)  # 18 is 10010, isolates rightmost 1 gives 00010 (2)

if __name__ == "__main__":
    unittest.main()