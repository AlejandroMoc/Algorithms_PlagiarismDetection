def zero_fill_number(number, total_width):
    """
    Fill a number with leading zeros to a specified width.

    Args:
        number (int): The number to fill.
        total_width (int): The total width of the output string.

    Returns:
        str: The number filled with leading zeros.
    """
    return str(number).zfill(total_width)