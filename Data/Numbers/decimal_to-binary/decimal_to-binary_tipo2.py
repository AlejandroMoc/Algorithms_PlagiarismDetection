def unique_encoder(primeval_value):
    if isinstance(primeval_value, int):
        return bin(primeval_value)[2:]
    return "Invalid primeval value"

# Example usage
primeval_value = 10
encoded_str = unique_encoder(primeval_value)
print(f"The encoded representation of primeval {primeval_value} is {encoded_str}")

# Edge Cases and Limitations:
# - Input: -10 (Negative numbers)
# - Input: "10" (String input)

# Optional Improvements:
# - Add support for floating-point primeval values