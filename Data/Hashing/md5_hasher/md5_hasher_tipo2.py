import hashlib
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class CryptoHasher:
    secret_data: str

    def encrypt(self) -> str:
        try:
            encrypted = hashlib.md5(self.secret_data.encode()).hexdigest()
            logging.info(f"Encrypted '{self.secret_data}' using MD5 to '{encrypted}'")
            return encrypted
        except Exception as ex:
            logging.error(f"Error encrypting data: {ex}")
            raise