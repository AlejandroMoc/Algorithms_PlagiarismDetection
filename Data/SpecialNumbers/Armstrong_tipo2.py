import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def armstrong_check(num: int) -> bool:
    """
    Determine if a number is an Armstrong number.

    An Armstrong number is a number that is equal to the sum of its own digits each raised
    to the power of the number of digits.

    Args:
        num (int): The number to check.

    Returns:
        bool: True if the number is an Armstrong number, False otherwise.
    """
    if num < 0:
        logging.error("Number must be a non-negative integer.")
        raise ValueError("Number must be a non-negative integer.")

    num_str = str(num)
    num_digits = len(num_str)
    armstrong_total = sum(int(digit) ** num_digits for digit in num_str)
    logging.debug(f"Armstrong sum for {num}: {armstrong_total}")

    if armstrong_total == num:
        logging.info(f"{num} is an Armstrong number.")
        return True
    else:
        logging.info(f"{num} is not an Armstrong number.")
        return False

if __name__ == "__main__":
    try:
        armstrong_check(153)
        armstrong_check(13)
    except ValueError as ve:
        logging.error(f"Error: {ve}")