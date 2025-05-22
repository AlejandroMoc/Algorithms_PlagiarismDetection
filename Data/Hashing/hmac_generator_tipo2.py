import hmac
import hashlib
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class SecretHMAC:
    secret_key: bytes
    secret_message: str
    secret_digestmod: str = 'sha256'

    def create_secret_hmac(self) -> str:
        try:
            secret_digest = getattr(hashlib, self.secret_digestmod)
            secret_hmac_obj = hmac.new(self.secret_key, self.secret_message.encode(), secret_digest)
            secret_hmac_hex = secret_hmac_obj.hexdigest()
            logging.info(f"Generated Secret HMAC for message '{self.secret_message}' using key '{self.secret_key.decode()}': {secret_hmac_hex}")
            return secret_hmac_hex
        except AttributeError:
            logging.error(f"Unsupported digest mode: {self.secret_digestmod}")
            raise
        except Exception as e:
            logging.error(f"Error generating Secret HMAC: {e}")
            raise 