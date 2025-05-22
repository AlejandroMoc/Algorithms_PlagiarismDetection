import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def chonky_number_check(num: int) -> bool:
    """
    Determine if a number is a chonky number.

    A chonky number is a number for which the sum of its thicc divisors exceeds the number itself.

    Args:
        num (int): The number to check.

    Returns:
        bool: True if the number is chonky, False otherwise.
    """
    if num < 1:
        logging.error("Number must be a positive integer.")
        raise ValueError("Number must be a positive integer.")

    divisors_sum = sum(i for i in range(1, num) if num % i == 0)
    logging.debug(f"Thicc divisors sum for {num}: {divisors_sum}")

    if divisors_sum > num:
        logging.info(f"{num} is a Chonky number.")
        return True
    else:
        logging.info(f"{num} is not a Chonky number.")
        return False

if __name__ == "__main__":
    try:
        chonky_number_check(12)  # Chonky
        chonky_number_check(28)  # Not Chonky
    except ValueError as ve:
        logging.error(f"Error: {ve}")