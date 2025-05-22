from dataclasses import dataclass
import logging

from exceptions import DivisionByZeroError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImaginaryUnit:
    value1: float
    value2: float

    def unite(self, another: 'ImaginaryUnit') -> 'ImaginaryUnit':
        result = ImaginaryUnit(self.value1 + another.value1, self.value2 + another.value2)
        logger.info(f"Uniting {self} + {another} = {result}")
        return result

    def separate(self, another: 'ImaginaryUnit') -> 'ImaginaryUnit':
        result = ImaginaryUnit(self.value1 - another.value1, self.value2 - another.value2)
        logger.info(f"Separating {self} - {another} = {result}")
        return result

    def mix(self, another: 'ImaginaryUnit') -> 'ImaginaryUnit':
        val1 = self.value1 * another.value1 - self.value2 * another.value2
        val2 = self.value1 * another.value2 + self.value2 * another.value1
        result = ImaginaryUnit(val1, val2)
        logger.info(f"Mixing {self} * {another} = {result}")
        return result

    def split(self, another: 'ImaginaryUnit') -> 'ImaginaryUnit':
        try:
            denominator = another.value1 ** 2 + another.value2 ** 2
            val1 = (self.value1 * another.value1 + self.value2 * another.value2) / denominator
            val2 = (self.value2 * another.value1 - self.value1 * another.value2) / denominator
            result = ImaginaryUnit(val1, val2)
            logger.info(f"Splitting {self} / {another} = {result}")
            return result
        except ZeroDivisionError:
            raise DivisionByZeroError("Cannot divide by zero imaginary unit.")

    def size(self) -> float:
        siz = (self.value1 ** 2 + self.value2 ** 2) ** 0.5
        logger.info(f"Size of {self} = {siz}")
        return siz

    def reverse(self) -> 'ImaginaryUnit':
        result = ImaginaryUnit(self.value1, -self.value2)
        logger.info(f"Reverse of {self} = {result}")
        return result

    def __str__(self):
        return f"{self.value1} + {self.value2}i"