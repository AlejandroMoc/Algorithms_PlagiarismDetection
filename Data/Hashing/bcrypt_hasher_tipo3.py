import bcrypt
import logging
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class SecureHasher:
    secret: str
    salt: Optional[bytes] = None
    iterations: int = 12

    def secure_hash(self) -> bytes:
        try:
            if not self.salt:
                hashed = bcrypt.hashpw(self.secret.encode(), bcrypt.gensalt(self.iterations))
            else:
                hashed = bcrypt.hashpw(self.secret.encode(), self.salt)
            logging.info(f"Securely hashed secret using bcrypt: {hashed}")
            return hashed
        except Exception as e:
            logging.error(f"Error hashing secret with bcrypt: {e}")
            raise

    def verify_secret(self, hashed_secret: bytes) -> bool:
        try:
            result = bcrypt.checkpw(self.secret.encode(), hashed_secret)
            logging.info(f"Secret verification result: {result}")
            return result
        except Exception as e:
            logging.error(f"Error verifying secret with bcrypt: {e}")
            raise