import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def is_happy_number(num: int) -> bool:
    """
    Determine if a number is a Happy number.

    A Happy number is a number defined by the process of replacing the number by the sum
    of the squares of its digits, repeating the process until the number equals 1 (where it will stay),
    or it loops endlessly in a cycle which does not include 1.

    Args:
        num (int): The number to check.

    Returns:
        bool: True if the number is happy, False otherwise.
    """
    if num < 1:
        logging.error("Number must be a positive integer.")
        raise ValueError("Number must be a positive integer.")

    seen_nums = set()
    original_num = num

    while num != 1 and num not in seen_nums:
        seen_nums.add(num)
        num = sum(int(digit) ** 2 for digit in str(num))
        logging.debug(f"Next number in sequence: {num}")

    if num == 1:
        logging.info(f"{original_num} is a Happy number.")
        return True
    else:
        logging.info(f"{original_num} is not a Happy number.")
        return False

if __name__ == "__main__":
    try:
        is_happy_number(19)
        is_happy_number(20)
    except ValueError as ve:
        logging.error(f"Error: {ve}")