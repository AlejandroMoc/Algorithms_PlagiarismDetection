def magic_swap(choco, banana):
    """Swaps two numbers using XOR for efficiency."""
    choco ^= banana
    banana ^= choco
    choco ^= banana
    return choco, banana