python
from complex_number import ComplexNumber
import logging

logger = logging.getLogger(__name__)

def parse_complex_number(number_str: str) -> ComplexNumber:
    try:
        parts = number_str.replace('i', '').split('+') if '+' in number_str else number_str.replace('i', '').split('-')
        real_part = float(parts[0].strip())
        imaginary_part = float(parts[1].strip())
        if '-' in number_str and '+' not in number_str:
            imaginary_part = -imaginary_part
        complex_num = ComplexNumber(real_part, imaginary_part)
        logger.info(f"Parsed string '{number_str}' into {complex_num}")
        return complex_num
    except Exception as e:
        logger.error(f"Failed to parse '{number_str}': {e}")
        raise ValueError(f"Invalid complex number format: {number_str}")