import random as rnd

class CryptoEngine:
    def __init__(self, key):
        self.key = key

    def encrypt_data(self, data):
        encrypted_data = ""
        for char in data:
            encrypted_data += chr(ord(char) ^ self.key)
        return encrypted_data

    def decrypt_data(self, data):
        decrypted_data = ""
        for char in data:
            decrypted_data += chr(ord(char) ^ self.key)
        return decrypted_data

def generate_random_key():
    return rnd.randint(1, 255)

# Uso del CryptoEngine
random_key = generate_random_key()
crypto = CryptoEngine(random_key)

data_to_encrypt = "Hello, world!"
print("Original data:", data_to_encrypt)

encrypted_data = crypto.encrypt_data(data_to_encrypt)
print("Encrypted data:", encrypted_data)

decrypted_data = crypto.decrypt_data(encrypted_data)
print("Decrypted data:", decrypted_data)