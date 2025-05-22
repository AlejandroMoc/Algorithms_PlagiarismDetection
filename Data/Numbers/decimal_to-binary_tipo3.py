def convert_decimal_to_binary(num):
    if isinstance(num, int):
        return bin(num)[2:]
    return "Invalid decimal number"

# Example usage
decimal_num = 10
binary_str = convert_decimal_to_binary(decimal_num)
print(f"The binary representation of decimal {decimal_num} is {binary_str}")

# Edge Cases and Limitations:
# - Input: -10 (Negative numbers)
# - Input: "10" (String input)

# Optional Improvements:
# - Add support for floating-point decimal numbers