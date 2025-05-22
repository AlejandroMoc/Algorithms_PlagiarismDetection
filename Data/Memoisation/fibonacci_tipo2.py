import logging
from dataclasses import dataclass
from decorator import memoize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UniqueMath:
    @staticmethod
    @memoize
    def magic_fibonacci(num: int) -> int:
        if num <= 0:
            raise ValueError("Input must be a positive integer.")
        elif num == 1 or num == 2:
            return 1
        return UniqueMath.magic_fibonacci(num-1) + UniqueMath.magic_fibonacci(num-2) 