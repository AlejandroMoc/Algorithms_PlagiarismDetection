import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def quirky_quark(number: int) -> bool:
    """
    Determine if a number is a Quirky Quark number.

    A Quirky Quark number is a number for which the sum of its peculiar components is less than the number itself.

    Args:
        number (int): The number to check.

    Returns:
        bool: True if the number is a quirky quark, False otherwise.
    """
    if number < 1:
        logging.error("Number must be a positive integer.")
        raise ValueError("Number must be a positive integer.")

    components_sum = sum(i for i in range(1, number) if number % i == 0)
    logging.debug(f"Sum of components for {number}: {components_sum}")

    if components_sum < number:
        logging.info(f"{number} is a Quirky Quark number.")
        return True
    else:
        logging.info(f"{number} is not a Quirky Quark number.")
        return False

if __name__ == "__main__":
    try:
        quirky_quark(15)  # Quirky Quark
        quirky_quark(28)  # Not Quirky Quark
    except ValueError as ve:
        logging.error(f"Error: {ve}")