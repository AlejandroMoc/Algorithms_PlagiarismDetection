import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryBank:
    store: Dict[Tuple, int] = field(default_factory=dict)
    
    def remember(self, operation):
        def wrapper(*args):
            try:
                if args in self.store:
                    logger.info(f"Recalling from memory for input: {args}")
                    return self.store[args]
                result = operation(*args)
                self.store[args] = result
                logger.info(f"Storing result for input: {args}")
                return result
            except Exception as e:
                logger.error(f"Error in MemoryBank: {e}")
                raise
        return wrapper

@dataclass
class MathGenius:
    memory_bank: MemoryBank = field(default_factory=MemoryBank)
    
    @staticmethod
    def calculate_combination(n: int, k: int) -> int:
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1
        return MathGenius.calculate_combination(n-1, k-1) + MathGenius.calculate_combination(n-1, k)
    
    def get_result(self, n: int, k: int) -> int:
        decorated_combination = self.memory_bank.remember(MathGenius.calculate_combination)
        return decorated_combination(n, k) 