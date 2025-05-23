import numpy as np

class CryptoMachine:
    def __init__(self, secret_key):
        self.secret_key = np.array(secret_key)
        self.modulus = 26
        self.validate_secret_key()

    def validate_secret_key(self):
        if self.secret_key.shape[0] != self.secret_key.shape[1]:
            raise ValueError("Secret key must be square")
        if np.linalg.det(self.secret_key) == 0:
            raise ValueError("Secret key must be invertible")

    def encode_data(self, text):
        text = text.upper().replace(" ", "")
        if len(text) % self.secret_key.shape[0] != 0:
            raise ValueError("Text length must be a multiple of secret key size")
        
        text_vector = [ord(char) - ord('A') for char in text]
        text_matrix = np.array(text_vector).reshape(-1, self.secret_key.shape[0])
        
        encoded_matrix = np.dot(text_matrix, self.secret_key) % self.modulus
        encoded_text = ''.join(chr(int(num) + ord('A')) for num in encoded_matrix.flatten())
        
        return encoded_text

    def decode_data(self, secret_text):
        secret_text = secret_text.upper().replace(" ", "")
        if len(secret_text) % self.secret_key.shape[0] != 0:
            raise ValueError("Secret text length must be a multiple of secret key size")
        
        secret_text_vector = [ord(char) - ord('A') for char in secret_text]
        secret_text_matrix = np.array(secret_text_vector).reshape(-1, self.secret_key.shape[0])
        
        inverse_secret_key = np.linalg.inv(self.secret_key)
        adjugate_matrix = np.round(inverse_secret_key * np.linalg.det(self.secret_key)).astype(int) % self.modulus
        determinant = int(np.round(np.linalg.det(self.secret_key))) % self.modulus
        determinant_inv = pow(determinant, -1, self.modulus)
        inverse_secret_key_mod = (determinant_inv * adjugate_matrix) % self.modulus
        
        decoded_matrix = np.dot(secret_text_matrix, inverse_secret_key_mod) % self.modulus
        decoded_text = ''.join(chr(int(num) + ord('A')) for num in decoded_matrix.flatten())
        
        return decoded_text