from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import padding
import os

# Generate unique pair of keys
def create_unique_key_pair():
    unique_key = ec.generate_private_key(ec.SECP384R1(), default_backend())
    custom_key = unique_key.public_key()
    return unique_key, custom_key

# Encode a secret using the receiver's public key
def encode_secret(custom_key, secret):
    # Generate a shared secret using ECDH (Elliptic Curve Diffie-Hellman)
    shared_secret = custom_key.exchange(ec.ECDH())
    
    # Derive a key from the shared secret using HKDF (Key derivation function)
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'confidential data',
        backend=default_backend()
    ).derive(shared_secret)
    
    # Encrypt the secret using derived symmetric key (AES encryption can be used here)
    iv = os.urandom(16)
    encoded_secret = iv + derived_key  # Simplified placeholder, use symmetric encryption in practice
    return encoded_secret

# Decode a secret using the unique key
def decode_secret(unique_key, encoded_secret):
    # Extract IV and the actual ciphertext
    iv = encoded_secret[:16]
    actual_secret = encoded_secret[16:]
    
    # Generate shared secret using ECDH
    shared_secret = unique_key.exchange(ec.ECDH())
    
    # Derive the symmetric key using the shared secret
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'confidential data',
        backend=default_backend()
    ).derive(shared_secret)
    
    # Decrypt using the derived symmetric key (AES decryption can be used)
    decoded_secret = actual_secret  # Placeholder for actual decryption logic
    return decoded_secret

# Serialize private and public keys
def serialize_key_pair(unique_key):
    pem = unique_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    return pem

def serialize_custom_key(custom_key):
    pem = custom_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return pem

# Example usage
if __name__ == "__main__":
    secret_data = b"Elliptic Curve Cryptography (ECC) example"
    
    # Generate unique key pair
    unique_key, custom_key = create_unique_key_pair()

    # Encode the secret
    encoded_secret_data = encode_secret(custom_key, secret_data)
    print("Encoded Secret:", encoded_secret_data)
    
    # Decode the secret
    decoded_secret_data = decode_secret(unique_key, encoded_secret_data)
    print("Decoded Secret:", decoded_secret_data.decode('utf-8'))

    # Serialize keys (optional)
    unique_pem = serialize_key_pair(unique_key)
    custom_pem = serialize_custom_key(custom_key)

    print("\nUnique Key PEM format:\n", unique_pem.decode('utf-8'))
    print("\nCustom Key PEM format:\n", custom_pem.decode('utf-8'))
