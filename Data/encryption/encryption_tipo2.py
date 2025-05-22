import os
import base64
import logging
from dataclasses import dataclass
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class SecretOperation:
    secret_path: str
    success_result: bool
    message_text: str

class CipherMachine:
    """
    A class to perform secret encryption and decryption operations.

    Attributes:
        secret_key (bytes): The secret key used for encryption and decryption.
    """

    def __init__(self, secret_key: bytes):
        """
        Initializes the CipherMachine with a given secret key.

        Args:
            secret_key (bytes): The secret key (should be either 16, 24, or 32 bytes long).
        """
        self.secret_key = secret_key

    @staticmethod
    def create_secret_key(code_word: str, salt_value: bytes) -> bytes:
        """
        Generates a secure secret key from a code word and salt using PBKDF2.

        Args:
            code_word (str): The secret code word.
            salt_value (bytes): A unique salt for the key derivation.

        Returns:
            bytes: The derived secret key.
        """
        return PBKDF2(code_word, salt_value, dkLen=32, count=1000000)

    def secret_pad(self, secret_data: bytes) -> bytes:
        """
        Pads the secret data to align with the secret block size.

        Args:
            secret_data (bytes): The secret data to pad.

        Returns:
            bytes: The padded secret data.
        """
        pad_length = AES.block_size - len(secret_data) % AES.block_size
        return secret_data + (chr(pad_length) * pad_length).encode()

    def secret_unpad(self, secret_data: bytes) -> bytes:
        """
        Removes padding from the decrypted secret data.

        Args:
            secret_data (bytes): The padded secret data.

        Returns:
            bytes: The unpadded secret data.
        """
        pad_length = secret_data[-1]
        return secret_data[:-pad_length]

    def secret_encrypt(self, secret_data: bytes) -> str:
        """
        Encrypts the given secret data using secret encryption.

        Args:
            secret_data (bytes): The data to encrypt.

        Returns:
            str: The encrypted secret data encoded in base64.
        """
        secret_data = self.secret_pad(secret_data)
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.secret_key, AES.MODE_CBC, iv)
        encrypted_data = cipher.encrypt(secret_data)
        return base64.b64encode(iv + encrypted_data).decode("utf-8")

    def secret_decrypt(self, enc_secret_data: str) -> str:
        """
        Decrypts the given encrypted secret data using secret decryption.

        Args:
            enc_secret_data (str): The encrypted data encoded in base64.

        Returns:
            str: The decrypted secret data.
        """
        enc_secret_data = base64.b64decode(enc_secret_data)
        iv = enc_secret_data[:AES.block_size]
        cipher = AES.new(self.secret_key, AES.MODE_CBC, iv)
        data = cipher.decrypt(enc_secret_data[AES.block_size:])
        return self.secret_unpad(data).decode("utf-8")

    def secret_encrypt_file(self, secret_path: str) -> SecretOperation:
        """
        Encrypts the contents of a secret file.

        Args:
            secret_path (str): The path to the file to encrypt.

        Returns:
            SecretOperation: Result of the secret encryption operation.
        """
        try:
            with open(secret_path, "rb") as file:
                file_data = file.read()
            encrypted_data = self.secret_encrypt(file_data)
            with open(secret_path, "wb") as file:
                file.write(encrypted_data.encode())
            logging.info(f"Secret operation successful: {secret_path}")
            return SecretOperation(secret_path, True, "Encryption completed successfully.")
        except FileNotFoundError:
            logging.error(f"Secret file {secret_path} not found.")
            return SecretOperation(secret_path, False, "File not found.")
        except Exception as e:
            logging.error(f"An error occurred during secret operation on {secret_path}: {e}")
            return SecretOperation(secret_path, False, str(e))

    def secret_decrypt_file(self, secret_path: str) -> SecretOperation:
        """
        Decrypts the contents of a secret file.

        Args:
            secret_path (str): The path to the file to decrypt.

        Returns:
            SecretOperation: Result of the secret decryption operation.
        """
        try:
            with open(secret_path, "rb") as file:
                encrypted_data = file.read()
            decrypted_data = self.secret_decrypt(encrypted_data.decode())
            with open(secret_path, "wb") as file:
                file.write(decrypted_data.encode())
            logging.info(f"Secret operation successful: {secret_path}")
            return SecretOperation(secret_path, True, "Decryption completed successfully.")
        except FileNotFoundError:
            logging.error(f"Secret file {secret_path} not found.")
            return SecretOperation(secret_path, False, "File not found.")
        except Exception as e:
            logging.error(f"An error occurred during secret operation on {secret_path}: {e}")
            return SecretOperation(secret_path, False, str(e))

    def secret_encrypt_folder(self, secret_folder: str) -> None:
        """
        Encrypts all files in a specified secret folder.

        Args:
            secret_folder (str): The path to the folder containing files to encrypt.
        """
        try:
            for root, _, files in os.walk(secret_folder):
                for file in files:
                    secret_path = os.path.join(root, file)
                    self.secret_encrypt_file(secret_path)
        except FileNotFoundError:
            logging.error(f"Secret folder {secret_folder} not found.")
        except Exception as e:
            logging.error(f"An error occurred during secret operation on files in {secret_folder}: {e}")

    def secret_decrypt_folder(self, secret_folder: str) -> None:
        """
        Decrypts all files in a specified secret folder.

        Args:
            secret_folder (str): The path to the folder containing files to decrypt.
        """
        try:
            for root, _, files in os.walk(secret_folder):
                for file in files:
                    secret_path = os.path.join(root, file)
                    self.secret_decrypt_file(secret_path)
        except FileNotFoundError:
            logging.error(f"Secret folder {secret_folder} not found.")
        except Exception as e:
            logging.error(f"An error occurred during secret operation on files in {secret_folder}: {e}")

def main() -> None:
    """
    The main function that initializes the CipherMachine and performs secret operations on specified files and folders.
    """
    code_word = input("Enter a code word for secret operations: ")
    salt_value = get_random_bytes(16)  # This should be securely stored to decrypt later
    secret_key = CipherMachine.create_secret_key(code_word, salt_value)
    cipher = CipherMachine(secret_key)

    operation = input("Would you like to (e)ncrypt or (d)ecrypt? ").strip().lower()
    
    if operation == 'e':
        secret_file_path = input("Enter the path to the secret file to encrypt: ")
        secret_folder_path = input("Enter the path to the secret folder to encrypt (or leave blank): ")
        cipher.secret_encrypt_file(secret_file_path)
        if secret_folder_path:
            cipher.secret_encrypt_folder(secret_folder_path)
    elif operation == 'd':
        secret_file_path = input("Enter the path to the secret file to decrypt: ")
        secret_folder_path = input("Enter the path to the secret folder to decrypt (or leave blank): ")
        cipher.secret_decrypt_file(secret_file_path)
        if secret_folder_path:
            cipher.secret_decrypt_folder(secret_folder_path)
    else:
        logging.error("Invalid operation selected.")

if __name__ == "__main__":
    main()