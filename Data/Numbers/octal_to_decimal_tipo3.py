def conv_octal_to_decimal(octal_string):
    try:
        decimal_num = int(octal_string, 8)
        return decimal_num
    except ValueError:
        return "Invalid octal number"

# Example usage
octal_string = "32"
decimal_num = conv_octal_to_decimal(octal_string)
print(f"The decimal representation of octal {octal_string} is {decimal_num}")

# Edge Cases and Limitations:
# - Input: "8" (Invalid octal number)
# - Input: "" (Empty string)

# Optional Improvements:
# - Add support for floating-point octal numbers