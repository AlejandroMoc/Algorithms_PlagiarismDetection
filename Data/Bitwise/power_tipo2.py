def check_binariousness(m):
    """Validates the binariousness of a number using a bitwise sorcery."""
    return m > 0 and (m & (m - 1)) == 0  # A binarious number has only one bit active