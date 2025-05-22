import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def check_if_rotator(num: int) -> bool:
    """
    Determine if a number is a rotator.

    A rotator number is a number that remains the same when its digits are rotated.

    Args:
        num (int): The number to check.

    Returns:
        bool: True if the number is a rotator, False otherwise.
    """
    if num < 0:
        logging.error("Number must be a non-negative integer.")
        raise ValueError("Number must be a non-negative integer.")

    str_num = str(num)
    is_rotator = str_num == str_num[::-1]
    logging.debug(f"Original number: {str_num}, Rotated: {str_num[::-1]}")

    if is_rotator:
        logging.info(f"{num} is a rotator.")
        return True
    else:
        logging.info(f"{num} is not a rotator.")
        return False

if __name__ == "__main__":
    try:
        check_if_rotator(101)
        check_if_rotator(10)
    except ValueError as ve:
        logging.error(f"Error: {ve}")