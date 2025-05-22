import unittest
from secret_cryptor import (
    hide_mystery, reveal_mystery,
    scramble_secret, unscramble_secret,
    encode_mountain, decode_mountain,
    disguise_secret_message, reveal_secret_message,
    lock_fence, unlock_fence
)

class TestSecretCryptor(unittest.TestCase):

    def test_mystery_hide(self):
        hidden_text = "PYSNIPPETS"
        a, b = 5, 8
        revealed_text = hide_mystery(hidden_text, a, b)
        original_text = reveal_mystery(revealed_text, a, b)
        self.assertEqual(original_text, hidden_text)

    def test_scramble_secret(self):
        secret_text = "PYSNIPPETS"
        shift = 3
        scrambled_text = scramble_secret(secret_text, shift)
        unscrambled_text = unscramble_secret(scrambled_text, shift)
        self.assertEqual(unscrambled_text, secret_text)

    def test_encode_mountain(self):
        message = "PYSNIP"
        key_matrix = [[6, 24, 1], [13, 16, 10], [20, 17, 15]]  # Example 3x3 matrix
        encoded_message = encode_mountain(message, key_matrix)
        decoded_message = decode_mountain(encoded_message, key_matrix)
        self.assertEqual(decoded_message, message)

    def test_disguise_secret_message(self):
        message = "PYSNIPPETS"
        key = "KEYWORD"
        disguised_message = disguise_secret_message(message, key)
        revealed_message = reveal_secret_message(disguised_message, key)
        self.assertEqual(revealed_message, message)

    def test_lock_fence(self):
        message = "PYSNIPPETS"
        key = 3
        locked_message = lock_fence(message, key)
        unlocked_message = unlock_fence(locked_message, key)
        self.assertEqual(unlocked_message, message)

    def test_vigenere_cipher(self):
        message = "PYSNIPPETS"
        key = "KEY"
        encoded_message = encode_mountain(message, key)
        decoded_message = decode_mountain(encoded_message, key)
        self.assertEqual(decoded_message, message)

if __name__ == "__main__":
    unittest.main()