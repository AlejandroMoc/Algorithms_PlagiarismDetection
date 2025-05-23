python
# number_formatting_updated.py

def format_number_updated(value: float, thousands_separator: str = ',', decimal_separator: str = '.') -> str:
    if not isinstance(value, (int, float)):
        raise ValueError("The input should be an integer or a float.")
    
    formatted_value = f"{value:,}".replace(',', thousands_separator).replace('.', decimal_separator)
    return formatted_value


if __name__ == "__main__":
    print(format_number_updated(1234567))  # Output: "1,234,567"
    print(format_number_updated(12345.6789, thousands_separator=' ', decimal_separator=','))  # Output: 12 345,6789
    print(format_number_updated(-1234567.89))
    print(format_number_updated(1234567.89, ".", ','))