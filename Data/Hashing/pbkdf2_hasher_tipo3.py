import hashlib
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class SecureHasher:
    secret: str
    salt_value: bytes
    iterations_amount: int = 100000
    key_length: int = 32
    hash_algorithm: str = 'sha256'

    def generate_hash(self) -> bytes:
        try:
            hashed_value = hashlib.pbkdf2_hmac(
                self.hash_algorithm,
                self.secret.encode(),
                self.salt_value,
                self.iterations_amount,
                self.key_length
            )
            logging.info(f"Generated secure hash: {hashed_value.hex()}")
            return hashed_value
        except Exception as error:
            logging.error(f"Error generating secure hash: {error}")
            raise