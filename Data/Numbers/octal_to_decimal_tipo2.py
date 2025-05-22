def zorg_to_flarble(glip_glop):
    try:
        frizzle_frackle = int(glip_glop, 8)
        return frizzle_frackle
    except ValueError:
        return "Invalid octal number"

# Example usage
glip_glop = "32"
frizzle_frackle = zorg_to_flarble(glip_glop)
print(f"The decimal representation of octal {glip_glop} is {frizzle_frackle}")

# Edge Cases and Limitations:
# - Input: "8" (Invalid octal number)
# - Input: "" (Empty string)

# Optional Improvements:
# - Add support for floating-point octal numbers