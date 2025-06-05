from affine_cipher import AffineCipher
from caesar_cipher import CaesarCipher
from hill_cipher import HillCipher
from playfair_cipher import playfair_encrypt, playfair_decrypt
from Rail_Fence_Cipher import rail_fence_cipher_encrypt, rail_fence_cipher_decrypt
from Vigenère_Cipher import VigenereCipher

def encrypt_affine(plain_text, a_val, b_val):
    cipher = AffineCipher(a_val, b_val)
    return cipher.encrypt(plain_text)

def decrypt_affine(cipher_text, a_val, b_val):
    cipher = AffineCipher(a_val, b_val)
    return cipher.decrypt(cipher_text)

def encrypt_caesar(plain_text, shift_val):
    cipher = CaesarCipher(shift_val)
    return cipher.encrypt(plain_text)

def decrypt_caesar(cipher_text, shift_val):
    cipher = CaesarCipher(shift_val)
    return cipher.decrypt(cipher_text)

def encrypt_hill(plain_text, key_matrix_val):
    cipher = HillCipher(key_matrix_val)
    return cipher.encrypt(plain_text)

def decrypt_hill(cipher_text, key_matrix_val):
    cipher = HillCipher(key_matrix_val)
    return cipher.decrypt(cipher_text)

def encrypt_playfair(plain_text, keyword):
    return playfair_encrypt(plain_text, keyword)

def decrypt_playfair(cipher_text, keyword):
    return playfair_decrypt(cipher_text, keyword)

def encrypt_rail_fence(plain_text, key_val):
    return rail_fence_cipher_encrypt(plain_text, key_val)

def decrypt_rail_fence(cipher_text, key_val):
    return rail_fence_cipher_decrypt(cipher_text, key_val)

def encrypt_vigenere(plain_text, keyword):
    cipher = VigenereCipher(keyword)
    return cipher.encrypt(plain_text)

def decrypt_vigenere(cipher_text, keyword):
    cipher = VigenereCipher(keyword)
    return cipher.decrypt(cipher_text)