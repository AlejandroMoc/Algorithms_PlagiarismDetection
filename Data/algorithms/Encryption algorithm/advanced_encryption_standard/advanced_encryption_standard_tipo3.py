from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os

def add_padding(data):
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()
    return padded_data

def remove_padding(data):
    unpadder = padding.PKCS7(128).unpadder()
    unpadded_data = unpadder.update(data) + unpadder.finalize()
    return unpadded_data

def encrypt_data(data, key):
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    padded_data = add_padding(data)
    
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return iv + ciphertext

def decrypt_data(ciphertext, key):
    iv = ciphertext[:16]
    actual_ciphertext = ciphertext[16:]
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()
    
    return remove_padding(padded_data)

if __name__ == "__main__":
    key = os.urandom(32)
    data = b"Advanced Encryption Standard (AES) example"
    
    encrypted_data = encrypt_data(data, key)
    print("Encrypted:", encrypted_data)
    
    decrypted_data = decrypt_data(encrypted_data, key)
    print("Decrypted:", decrypted_data.decode('utf-8'))