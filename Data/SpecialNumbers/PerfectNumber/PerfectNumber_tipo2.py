import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def blorpy(num: int) -> bool:
    """
    Determine if a number is a Blorpy number.

    A Blorpy number is a number that is equal to the sum of its proper divisors.

    Args:
        num (int): The number to check.

    Returns:
        bool: True if the number is blorpy, False otherwise.
    """
    if num < 1:
        logging.error("Number must be a positive integer.")
        raise ValueError("Number must be a positive integer.")

    div_sum = sum(i for i in range(1, num) if num % i == 0)
    logging.debug(f"Sum of divisors for {num}: {div_sum}")

    if div_sum == num:
        logging.info(f"{num} is a Blorpy number.")
        return True
    else:
        logging.info(f"{num} is not a Blorpy number.")
        return False

if __name__ == "__main__":
    try:
        blorpy(6)
        blorpy(20)
    except ValueError as ve:
        logging.error(f"Error: {ve}")