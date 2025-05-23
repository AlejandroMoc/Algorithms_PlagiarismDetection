import bcrypt
import logging
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class CrypticHasher:
    secret: str
    token: Optional[bytes] = None
    cycles: int = 12

    def encode_secret(self) -> bytes:
        try:
            if not self.token:
                encrypted = bcrypt.hashpw(self.secret.encode(), bcrypt.gensalt(self.cycles))
            else:
                encrypted = bcrypt.hashpw(self.secret.encode(), self.token)
            logging.info(f"Encoded secret using bcrypt: {encrypted}")
            return encrypted
        except Exception as e:
            logging.error(f"Error encoding secret with bcrypt: {e}")
            raise

    def validate_secret(self, encrypted: bytes) -> bool:
        try:
            result = bcrypt.checkpw(self.secret.encode(), encrypted)
            logging.info(f"Secret validation result: {result}")
            return result
        except Exception as e:
            logging.error(f"Error validating secret with bcrypt: {e}")
            raise 