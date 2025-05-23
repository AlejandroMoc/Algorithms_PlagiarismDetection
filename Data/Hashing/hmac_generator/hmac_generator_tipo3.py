import hmac
import hashlib
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class HMACGenerator:
    secret_key: bytes
    input_message: str
    hash_algorithm: str = 'sha256'

    def generate_hmac(self) -> str:
        try:
            hash_function = getattr(hashlib, self.hash_algorithm)
            hmac_object = hmac.new(self.secret_key, self.input_message.encode(), hash_function)
            hmac_hex = hmac_object.hexdigest()
            logging.info(f"Generated HMAC for message '{self.input_message}' using key '{self.secret_key.decode()}': {hmac_hex}")
            return hmac_hex
        except AttributeError:
            logging.error(f"Unsupported hash algorithm: {self.hash_algorithm}")
            raise
        except Exception as error:
            logging.error(f"Error generating HMAC: {error}")
            raise