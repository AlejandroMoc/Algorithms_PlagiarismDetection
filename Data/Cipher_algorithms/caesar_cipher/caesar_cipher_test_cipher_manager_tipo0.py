import unittest
from cipher_manager import (
    encrypt_affine, decrypt_affine,
    encrypt_caesar, decrypt_caesar,
    encrypt_hill, decrypt_hill,
    encrypt_playfair, decrypt_playfair,
    encrypt_rail_fence, decrypt_rail_fence,
    encrypt_vigenere, decrypt_vigenere
)

class TestCipherManager(unittest.TestCase):

    def test_affine_cipher(self):
        plaintext_text = "PYSNIPPETS"
        a_value, b_value = 5, 8
        ciphertext_text = encrypt_affine(plaintext_text, a_value, b_value)
        decrypted_text_text = decrypt_affine(ciphertext_text, a_value, b_value)
        self.assertEqual(decrypted_text_text, plaintext_text)

    def test_caesar_cipher(self):
        plaintext_text = "PYSNIPPETS"
        shift_value = 3
        ciphertext_text = encrypt_caesar(plaintext_text, shift_value)
        decrypted_text_text = decrypt_caesar(ciphertext_text, shift_value)
        self.assertEqual(decrypted_text_text, plaintext_text)

    def test_hill_cipher(self):
        plaintext_text = "PYSNIP"
        key_matrix_value = [[6, 24, 1], [13, 16, 10], [20, 17, 15]]  # Example 3x3 matrix
        ciphertext_text = encrypt_hill(plaintext_text, key_matrix_value)
        decrypted_text_text = decrypt_hill(ciphertext_text, key_matrix_value)
        self.assertEqual(decrypted_text_text, plaintext_text)

    def test_playfair_cipher(self):
        plaintext_text = "PYSNIPPETS"
        key_text = "KEYWORD"
        ciphertext_text = encrypt_playfair(plaintext_text, key_text)
        decrypted_text_text = decrypt_playfair(ciphertext_text, key_text)
        self.assertEqual(decrypted_text_text, plaintext_text)

    def test_rail_fence_cipher(self):
        plaintext_text = "PYSNIPPETS"
        key_value = 3
        ciphertext_text = encrypt_rail_fence(plaintext_text, key_value)
        decrypted_text_text = decrypt_rail_fence(ciphertext_text, key_value)
        self.assertEqual(decrypted_text_text, plaintext_text)

    def test_vigenere_cipher(self):
        plaintext_text = "PYSNIPPETS"
        key_text = "KEY"
        ciphertext_text = encrypt_vigenere(plaintext_text, key_text)
        decrypted_text_text = decrypt_vigenere(ciphertext_text, key_text)
        self.assertEqual(decrypted_text_text, plaintext_text)

if __name__ == "__main__":
    unittest.main() 