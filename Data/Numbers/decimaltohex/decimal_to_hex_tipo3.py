def convert_decimal_to_hex(decimal_val):
    if isinstance(decimal_val, int):
        return hex(decimal_val)[2:].upper()
    return "Invalid decimal number"

# Example usage
decimal_val = 26
hex_string = convert_decimal_to_hex(decimal_val)
print(f"The hexadecimal representation of decimal {decimal_val} is {hex_string}")

# Edge Cases and Limitations:
# - Input: -26 (Negative numbers)
# - Input: "26" (String input)

# Optional Improvements:
# - Add support for floating-point decimal numbers