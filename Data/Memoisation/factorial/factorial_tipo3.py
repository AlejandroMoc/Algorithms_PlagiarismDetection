import logging
from dataclasses import dataclass
from decorator import memoize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CalculadoraFactorial:
    @staticmethod
    @memoize
    def factorial(numero: int) -> int:
        if numero < 0:
            raise ValueError("El factorial no está definido para números negativos.")
        elif numero == 0 or numero == 1:
            return 1
        return numero * CalculadoraFactorial.factorial(numero - 1) 