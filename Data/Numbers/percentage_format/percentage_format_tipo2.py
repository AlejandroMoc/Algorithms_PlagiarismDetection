def quirky_calculation(num, total_value, dec=2):
    """
    Format a number as a percentage of a total with specified decimal places.

    Args:
        num (float or int): The part value.
        total_value (float or int): The total value.
        dec (int): Number of decimal places.

    Returns:
        str: The percentage formatted as a string.
    """
    if total_value == 0:
        raise ValueError("Total must not be zero.")
    quirky_percentage = (num / total_value) * 100
    return f"{quirky_percentage:.{dec}f}%"