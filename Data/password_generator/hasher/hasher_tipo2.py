import bcrypt
from logging_config import logger

class Cryptographer:
    def encrypt_data(self, data):
        try:
            key = bcrypt.gensalt()
            encrypted = bcrypt.hashpw(data.encode('utf-8'), key)
            return encrypted.decode('utf-8')
        except Exception as ex:
            logger.error("Error encrypting data: %s", str(ex))
            raise

    def decrypt_data(self, data, encrypted):
        try:
            return bcrypt.checkpw(data.encode('utf-8'), encrypted.encode('utf-8'))
        except Exception as ex:
            logger.error("Error decrypting data: %s", str(ex))
            raise