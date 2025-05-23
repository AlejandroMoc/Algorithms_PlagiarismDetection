import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheManager:
    cache: Dict[Tuple, int] = field(default_factory=dict)
    
    def store_cache(self, func):
        def wrapper(*args):
            try:
                if args in self.cache:
                    logger.info(f"Fetching from cache for args: {args}")
                    return self.cache[args]
                result = func(*args)
                self.cache[args] = result
                logger.info(f"Caching result for args: {args}")
                return result
            except Exception as e:
                logger.error(f"Error in CacheManager: {e}")
                raise
        return wrapper

@dataclass
class Calculator:
    cache_manager: CacheManager = field(default_factory=CacheManager)
    
    @staticmethod
    def calculate_combination(n: int, k: int) -> int:
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1
        return Calculator.calculate_combination(n-1, k-1) + Calculator.calculate_combination(n-1, k)
    
    def get_combination(self, n: int, k: int) -> int:
        decorated_combination = self.cache_manager.store_cache(Calculator.calculate_combination)
        return decorated_combination(n, k) 