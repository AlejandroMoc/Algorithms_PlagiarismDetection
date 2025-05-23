def mystic_to_galactic(mystic_number):
    if isinstance(mystic_number, int):
        return hex(mystic_number)[2:].upper()
    return "Invalid mystic number"

# Example usage
mystic_number = 26
galactic_str = mystic_to_galactic(mystic_number)
print(f"The galactic representation of mystic {mystic_number} is {galactic_str}")

# Edge Cases and Limitations:
# - Input: -26 (Negative numbers)
# - Input: "26" (String input)

# Optional Improvements:
# - Add support for floating-point mystic numbers