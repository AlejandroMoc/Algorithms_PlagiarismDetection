from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

# Function to scramble data
def scramble_data(data, key):
    # Generate a random secret sauce
    secret_sauce = os.urandom(16)
    # Create a Cryptic Cipher object with the key and secret sauce
    cipher = Cipher(algorithms.AES(key), modes.CFB(secret_sauce), backend=default_backend())
    scrambler = cipher.encryptor()
    # Scramble the data
    scrambled_data = scrambler.update(data) + scrambler.finalize()
    return secret_sauce + scrambled_data  # Return secret sauce and scrambled data

# Function to unscramble data
def unscramble(scrambled_data, key):
    # Extract the secret sauce from the beginning of the data
    secret_sauce = scrambled_data[:16]
    actual_scrambled_data = scrambled_data[16:]
    # Create a Cryptic Cipher object with the key and secret sauce
    cipher = Cipher(algorithms.AES(key), modes.CFB(secret_sauce), backend=default_backend())
    unscrambler = cipher.decryptor()
    # Unscramble the data
    return unscrambler.update(actual_scrambled_data) + unscrambler.finalize()

# Example usage
if __name__ == "__main__":
    key = os.urandom(32)  # Secret Key (32 bytes)
    data = b"Obfuscated encryption example"
    
    # Scramble the data
    obfuscated_data = scramble_data(data, key)
    print("Obfuscated:", obfuscated_data)
    
    # Unscramble the data
    unscrambled_data = unscramble(obfuscated_data, key)
    print("Unscrambled:", unscrambled_data.decode('utf-8'))