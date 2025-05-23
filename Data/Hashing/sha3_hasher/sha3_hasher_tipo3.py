import hashlib
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class HashingAlgorithm:
    data: str
    variant: str = 'sha3_256'

    def hash_data(self) -> str:
        try:
            hash_func = getattr(hashlib, self.variant)
            hashed_data = hash_func(self.data.encode()).hexdigest()
            logging.info(f"Data hashed using {self.variant}: {hashed_data}")
            return hashed_data
        except AttributeError:
            logging.error(f"Unsupported hashing algorithm: {self.variant}")
            raise
        except Exception as e:
            logging.error(f"Error hashing data: {e}")
            raise