def xor_magic(array):
    """Finds the mystical number in a list where every other number appears twice using XOR for efficiency."""
    mystical_number = 0
    for number in array:
        mystical_number ^= number  # XOR all numbers; duplicates will cancel out
    return mystical_number