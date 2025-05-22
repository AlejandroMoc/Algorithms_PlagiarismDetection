def extract_singular(n):
    """Extracts the singular rightmost bit in n using bitwise operations."""
    return n & -n  # Employ two's complement to extract the rightmost singular bit