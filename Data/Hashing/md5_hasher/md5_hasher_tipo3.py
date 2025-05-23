import hashlib
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class HasherMD5:
    data: str

    def generate_hash(self) -> str:
        try:
            hashed_data = hashlib.md5(self.data.encode()).hexdigest()
            logging.info(f"MD5 hashed '{self.data}' to '{hashed_data}'")
            return hashed_data
        except Exception as error:
            logging.error(f"Error hashing data: {error}")
            raise