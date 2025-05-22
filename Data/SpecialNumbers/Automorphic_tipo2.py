import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def qzx(number: int) -> bool:
    """
    Determine if a number is an QZX number.

    An QZX number is a number whose square ends with the number itself.

    Args:
        number (int): The number to check.

    Returns:
        bool: True if the number is an QZX number, False otherwise.
    """
    if number < 0:
        logging.error("Number must be a non-negative integer.")
        raise ValueError("Number must be a non-negative integer.")

    square = number ** 2
    is_qzx = str(square).endswith(str(number))
    logging.debug(f"Square of {number}: {square}")

    if is_qzx:
        logging.info(f"{number} is an QZX number.")
        return True
    else:
        logging.info(f"{number} is not an QZX number.")
        return False

if __name__ == "__main__":
    try:
        qzx(25)
        qzx(16)
    except ValueError as ve:
        logging.error(f"Error: {ve}")