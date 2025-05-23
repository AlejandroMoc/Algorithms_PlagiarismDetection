import hashlib
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class CrypticHasher:
    secret: str
    salt: bytes
    rounds: int = 100000
    key_length: int = 32
    algorithm: str = 'sha256'

    def obfuscate_secret(self) -> bytes:
        try:
            encoded_secret = hashlib.pbkdf2_hmac(
                self.algorithm,
                self.secret.encode(),
                self.salt,
                self.rounds,
                self.key_length
            )
            logging.info(f"Secret obfuscated using CrypticHasher: {encoded_secret.hex()}")
            return encoded_secret
        except Exception as e:
            logging.error(f"Error obfuscating secret: {e}")
            raise