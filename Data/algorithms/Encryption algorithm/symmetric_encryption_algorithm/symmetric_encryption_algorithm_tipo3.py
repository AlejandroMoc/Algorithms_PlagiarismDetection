from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

def encrypt_data(data, key):
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(data) + encryptor.finalize()
    return iv + ciphertext

def decrypt_data(ciphertext, key):
    iv = ciphertext[:16]
    actual_ciphertext = ciphertext[16:]
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(actual_ciphertext) + decryptor.finalize()

if __name__ == "__main__":
    key = os.urandom(32)
    data = b"Symmetric encryption example"
    
    encrypted_data = encrypt_data(data, key)
    print("Encrypted:", encrypted_data)
    
    decrypted_data = decrypt_data(encrypted_data, key)
    print("Decrypted:", decrypted_data.decode('utf-8'))