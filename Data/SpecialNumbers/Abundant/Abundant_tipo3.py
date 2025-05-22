import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def check_abundance(num: int) -> bool:
    if num < 1:
        logging.error("Number must be a positive integer.")
        raise ValueError("Number must be a positive integer.")

    div_sum = sum(i for i in range(1, num) if num % i == 0)
    logging.debug(f"Sum of divisors for {num}: {div_sum}")

    if div_sum > num:
        logging.info(f"{num} is an Abundant number.")
        return True
    else:
        logging.info(f"{num} is not an Abundant number.")
        return False

if __name__ == "__main__":
    try:
        check_abundance(12)  # Abundant
        check_abundance(28)  # Not Abundant
    except ValueError as error:
        logging.error(f"Error: {error}")