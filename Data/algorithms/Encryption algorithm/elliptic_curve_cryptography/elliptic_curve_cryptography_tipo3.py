from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import padding
import os

def gen_key_pair_ecc():
    private_key = ec.generate_private_key(ec.SECP384R1(), default_backend())
    public_key = private_key.public_key()
    return private_key, public_key

def encrypt_msg(public_key, msg):
    shared_key = public_key.exchange(ec.ECDH())
    
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'handshake data',
        backend=default_backend()
    ).derive(shared_key)
    
    iv = os.urandom(16)
    ciphertext = iv + derived_key
    return ciphertext

def decrypt_msg(private_key, ciphertext):
    iv = ciphertext[:16]
    actual_ciphertext = ciphertext[16:]
    
    shared_key = private_key.exchange(ec.ECDH())
    
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'handshake data',
        backend=default_backend()
    ).derive(shared_key)
    
    decrypted_msg = actual_ciphertext
    return decrypted_msg

def serialize_private(private_key):
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    return pem

def serialize_public(public_key):
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return pem

if __name__ == "__main__":
    message = b"Elliptic Curve Cryptography (ECC) example"
    
    private_key, public_key = gen_key_pair_ecc()

    encrypted_message = encrypt_msg(public_key, message)
    print("Encrypted:", encrypted_message)
    
    decrypted_message = decrypt_msg(private_key, encrypted_message)
    print("Decrypted:", decrypted_message.decode('utf-8'))

    private_pem = serialize_private(private_key)
    public_pem = serialize_public(public_key)

    print("\nPrivate Key PEM format:\n", private_pem.decode('utf-8'))
    print("\nPublic Key PEM format:\n", public_pem.decode('utf-8'))