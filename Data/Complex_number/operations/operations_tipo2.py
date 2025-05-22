from custom_math_object import FancyNumber
from log_utils import configure_logging

log_handler = configure_logging(__name__)

def calculate_power(fancy_num: FancyNumber, exp_value: int) -> FancyNumber:
    if exp_value == 0:
        log_handler.info(f"Special Power: {fancy_num} ** {exp_value} = 1 + 0i")
        return FancyNumber(1, 0)
    if exp_value < 0:
        log_handler.error("Unsupported negative exponents.")
        raise ValueError("Unsupported negative exponents.")
    try:
        result = FancyNumber(1, 0)  # Starting result as 1 + 0i
        for _ in range(exp_value):
            result = result.multiply(fancy_num)
        log_handler.info(f"Special Power: {fancy_num} ** {exp_value} = {result}")
        return result
    except Exception as e:
        log_handler.error(f"Power computation error: {e}")
        raise ValueError("Invalid exponent value.")