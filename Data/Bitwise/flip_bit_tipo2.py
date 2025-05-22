def binary_flip(num, position):
    """Flips the bit at the specified position in num using bitwise XOR."""
    return num ^ (1 << position)  # Flip the bit at position position