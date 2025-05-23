# custom_formatting.py

def numerical_format(val: float, sep1: str = ',', sep2: str = '.') -> str:
    """
    Format a number with custom separators.

    Args:
        val (int, float): The number to format.

    Returns:
        str: The formatted number as a string.
    """
    if not isinstance(val, (int, float)):
        raise ValueError("The input should be an integer or a float.")
    
    formatted_val = f"{val:,}".replace(',', sep1).replace('.', sep2)
    return formatted_val


# Example usage
if __name__ == "__main__":
    print(numerical_format(1234567)) # Output: "1,234,567"
    print(numerical_format(12345.6789, sep1=' ', sep2=',')) #output: 12 345,6789
    print(numerical_format(-1234567.89))
    print(numerical_format(1234567.89, ".", ','))