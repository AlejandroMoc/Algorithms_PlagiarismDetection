import numpy as np

class HillCipher:
    def __init__(self, key):
        self.key = np.array(key)
        self.mod = 26
        self.validate_key()

    def validate_key(self):
        if self.key.shape[0] != self.key.shape[1]:
            raise ValueError("Key matrix must be square")
        if np.linalg.det(self.key) == 0:
            raise ValueError("Key matrix must be invertible")

    def encrypt(self, text):
        text = text.upper().replace(" ", "")
        if len(text) % self.key.shape[0] != 0:
            raise ValueError("Text length must be a multiple of key matrix size")
        
        text_vector = [ord(char) - ord('A') for char in text]
        text_matrix = np.array(text_vector).reshape(-1, self.key.shape[0])
        
        encrypted_matrix = np.dot(text_matrix, self.key) % self.mod
        encrypted_text = ''.join(chr(int(num) + ord('A')) for num in encrypted_matrix.flatten())
        
        return encrypted_text

    def decrypt(self, text):
        text = text.upper().replace(" ", "")
        if len(text) % self.key.shape[0] != 0:
            raise ValueError("Text length must be a multiple of key matrix size")
        
        text_vector = [ord(char) - ord('A') for char in text]
        text_matrix = np.array(text_vector).reshape(-1, self.key.shape[0])
        
        inv_key = np.linalg.inv(self.key)
        adj_matrix = np.round(inv_key * np.linalg.det(self.key)).astype(int) % self.mod
        det = int(np.round(np.linalg.det(self.key))) % self.mod
        det_inv = pow(det, -1, self.mod)
        inv_key_mod = (det_inv * adj_matrix) % self.mod
        
        decrypted_matrix = np.dot(text_matrix, inv_key_mod) % self.mod
        decrypted_text = ''.join(chr(int(num) + ord('A')) for num in decrypted_matrix.flatten())
        
        return decrypted_text