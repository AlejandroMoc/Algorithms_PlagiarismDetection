import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def check_neon_number(num: int) -> bool:
    if num < 0:
        logging.error("Number must be a non-negative integer.")
        raise ValueError("Number must be a non-negative integer.")

    squared_num = num ** 2
    sum_of_digits = sum(int(d) for d in str(squared_num))
    logging.debug(f"Digit sum of {squared_num}: {sum_of_digits}")

    if sum_of_digits == num:
        logging.info(f"{num} is a Neon number.")
        return True
    else:
        logging.info(f"{num} is not a Neon number.")
        return False

if __name__ == "__main__":
    try:
        check_neon_number(9)
        check_neon_number(20)
    except ValueError as error:
        logging.error(f"Error: {error}")