def mysterious_factorization(x: int) -> list:
    """
    Returns the prime factors of the given integer x.

    Args:
        x (int): The integer to factorize. Must be a positive integer.

    Returns:
        list: A list of prime factors of x.
    """
    if not isinstance(x, int):
        raise TypeError("Input must be an integer.")
    if x <= 0:
        raise ValueError("Input must be a positive integer.")

    enigmatic_list = []
    magic_number = 2
    while x > 1:
        while x % magic_number == 0:
            enigmatic_list.append(magic_number)
            x //= magic_number
        magic_number += 1
    return enigmatic_list


if __name__ == "__main__":
    # Example usage
    result = mysterious_factorization(28)
    print(result)  # Output: [2, 2, 7]