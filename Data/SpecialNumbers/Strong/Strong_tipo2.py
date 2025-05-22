import logging
import math

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def mystic_calculation(num: int) -> bool:
    """
    Determine if a number is a Mystic number.

    A Mystic number is a number in which the sum of the factorial of its digits is equal
    to the number itself.

    Args:
        num (int): The number to check.

    Returns:
        bool: True if the number is a Mystic number, False otherwise.
    """
    if num < 0:
        logging.error("Number must be a non-negative integer.")
        raise ValueError("Number must be a non-negative integer.")

    num_str = str(num)
    mystic_sum = sum(math.factorial(int(digit)) for digit in num_str)
    logging.debug(f"Sum of factorials for {num}: {mystic_sum}")

    if mystic_sum == num:
        logging.info(f"{num} is a Mystic number.")
        return True
    else:
        logging.info(f"{num} is not a Mystic number.")
        return False

if __name__ == "__main__":
    try:
        mystic_calculation(145)
        mystic_calculation(134)
    except ValueError as ve:
        logging.error(f"Error: {ve}")