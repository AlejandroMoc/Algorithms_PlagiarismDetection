from affine_cipher import AffineCipher
from caesar_cipher import CaesarCipher
from hill_cipher import HillCipher
from playfair_cipher import playfair_encrypt, playfair_decrypt
from Rail_Fence_Cipher import rail_fence_cipher_encrypt, rail_fence_cipher_decrypt
from Vigenère_Cipher import VigenereCipher

def encode_affine(text, a, b):
    secret = AffineCipher(a, b)
    return secret.encrypt(text)

def decode_affine(secret_text, a, b):
    secret = AffineCipher(a, b)
    return secret.decrypt(secret_text)

def encode_caesar(text, shift):
    secret = CaesarCipher(shift)
    return secret.encrypt(text)

def decode_caesar(secret_text, shift):
    secret = CaesarCipher(shift)
    return secret.decrypt(secret_text)

def encode_hill(text, key_matrix):
    secret = HillCipher(key_matrix)
    return secret.encrypt(text)

def decode_hill(secret_text, key_matrix):
    secret = HillCipher(key_matrix)
    return secret.decrypt(secret_text)

def encode_playfair(text, key):
    return playfair_encrypt(text, key)

def decode_playfair(secret_text, key):
    return playfair_decrypt(secret_text, key)

def encode_rail_fence(text, key):
    return rail_fence_cipher_encrypt(text, key)

def decode_rail_fence(secret_text, key):
    return rail_fence_cipher_decrypt(secret_text, key)

def encode_vigenere(text, key):
    secret = VigenereCipher(key)
    return secret.encrypt(text)

def decode_vigenere(secret_text, key):
    secret = VigenereCipher(key)
    return secret.decrypt(secret_text)