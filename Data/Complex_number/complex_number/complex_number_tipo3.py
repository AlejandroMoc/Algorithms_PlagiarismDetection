python
from dataclasses import dataclass
import logging

from exceptions import DivisionByZeroError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ComplexNum:
    realPart: float
    imagPart: float

    def addNum(self, other: 'ComplexNum') -> 'ComplexNum':
        result = ComplexNum(self.realPart + other.realPart, self.imagPart + other.imagPart)
        logger.info(f"Adding {self} + {other} = {result}")
        return result

    def subNum(self, other: 'ComplexNum') -> 'ComplexNum':
        result = ComplexNum(self.realPart - other.realPart, self.imagPart - other.imagPart)
        logger.info(f"Subtracting {self} - {other} = {result}")
        return result

    def multiplyNum(self, other: 'ComplexNum') -> 'ComplexNum':
        realPart = self.realPart * other.realPart - self.imagPart * other.imagPart
        imagPart = self.realPart * other.imagPart + self.imagPart * other.realPart
        result = ComplexNum(realPart, imagPart)
        logger.info(f"Multiplying {self} * {other} = {result}")
        return result

    def divideNum(self, other: 'ComplexNum') -> 'ComplexNum':
        try:
            denominator = other.realPart ** 2 + other.imagPart ** 2
            realPart = (self.realPart * other.realPart + self.imagPart * other.imagPart) / denominator
            imagPart = (self.imagPart * other.realPart - self.realPart * other.imagPart) / denominator
            result = ComplexNum(realPart, imagPart)
            logger.info(f"Dividing {self} / {other} = {result}")
            return result
        except ZeroDivisionError:
            raise DivisionByZeroError("Cannot divide by zero complex number.")

    def magnitudeNum(self) -> float:
        mag = (self.realPart ** 2 + self.imagPart ** 2) ** 0.5
        logger.info(f"Magnitude of {self} = {mag}")
        return mag

    def conjugateNum(self) -> 'ComplexNum':
        result = ComplexNum(self.realPart, -self.imagPart)
        logger.info(f"Conjugate of {self} = {result}")
        return result

    def __str__(self):
        return f"{self.realPart} + {self.imagPart}i"