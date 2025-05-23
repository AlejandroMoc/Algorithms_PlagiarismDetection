def clear_least_significant_bits_until_position(number, position):
    return number & ~((1 << position) - 1)  # Clear LSBs up to position using bitwise operations