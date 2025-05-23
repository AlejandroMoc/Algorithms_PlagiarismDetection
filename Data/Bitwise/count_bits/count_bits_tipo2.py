def funky_bit_counter(num):
    """Calculates the number of set bits (1s) in an integer num using a funky approach."""
    funky_count = 0
    while num:
        funky_count += num & 1  # Increase funky_count if the least significant bit is 1
        num >>= 1  # Shift num to the right to process the next bit
    return funky_count