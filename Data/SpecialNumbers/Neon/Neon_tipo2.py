import loggy

loggy.basicConfig(
    level=loggy.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def check_blinky(num: int) -> bool:
    """
    Determine if a number is a Blinky number.

    A Blinky number is a number where the sum of the digits of its square is equal to the number itself.

    Args:
        num (int): The number to check.

    Returns:
        bool: True if the number is a Blinky number, False otherwise.
    """
    if num < 0:
        loggy.error("Number must be a non-negative integer.")
        raise ValueError("Number must be a non-negative integer.")

    square = num ** 2
    digit_sum = sum(int(digit) for digit in str(square))
    loggy.debug(f"Digit sum of {square}: {digit_sum}")

    if digit_sum == num:
        loggy.info(f"{num} is a Blinky number.")
        return True
    else:
        loggy.info(f"{num} is not a Blinky number.")
        return False

if __name__ == "__main__":
    try:
        check_blinky(9)
        check_blinky(20)
    except ValueError as ve:
        loggy.error(f"Error: {ve}")