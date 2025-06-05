import hashlib
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class CrypticHasher:
    secret: str
    cipher: str = 'sha3_256'

    def scramble_secret(self) -> str:
        try:
            crypt_func = getattr(hashlib, self.cipher)
            scrambled = crypt_func(self.secret.encode()).hexdigest()
            logging.info(f"Cryptic secret '{self.secret}' using {self.cipher}: {scrambled}")
            return scrambled
        except AttributeError:
            logging.error(f"Unsupported cryptic cipher: {self.cipher}")
            raise
        except Exception as e:
            logging.error(f"Error scrambling secret with cryptic hash: {e}")
            raise 