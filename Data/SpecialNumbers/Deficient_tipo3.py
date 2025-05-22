import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def is_deficient_number(num: int) -> bool:
    if num < 1:
        logging.error("Number must be a positive integer.")
        raise ValueError("Number must be a positive integer.")

    divisors_sum = sum(i for i in range(1, num) if num % i == 0)
    logging.debug(f"Sum of divisors for {num}: {divisors_sum}")

    if divisors_sum < num:
        logging.info(f"{num} is a Deficient number.")
        return True
    else:
        logging.info(f"{num} is not a Deficient number.")
        return False

if __name__ == "__main__":
    try:
        is_deficient_number(15)  # Deficient
        is_deficient_number(28)  # Not Deficient
    except ValueError as error:
        logging.error(f"Error: {error}")