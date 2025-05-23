from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

# Generate RSA key pair
def create_key_pair():
    secret_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_secret = secret_key.public_key()
    return secret_key, public_secret

# Encrypt data using public key
def obfuscate(public_secret, msg):
    ciphertext = public_secret.encrypt(
        msg,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return ciphertext

# Decrypt data using private key
def deobfuscate(secret_key, ciphertext):
    plaintext = secret_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return plaintext

# Serialize the private key to save it to a file
def store_secret_key(secret_key, pswd=None):
    pem_format = secret_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.BestAvailableEncryption(pswd) if pswd else serialization.NoEncryption()
    )
    return pem_format

# Serialize the public key to save it to a file
def store_public_key(public_secret):
    pem_format = public_secret.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return pem_format

# Example usage
if __name__ == "__main__":
    msg = b"Asymmetric encryption example with RSA"
    
    # Generate RSA key pair
    secret_key, public_secret = create_key_pair()

    # Encrypt the message
    ciphertext = obfuscate(public_secret, msg)
    print("Encrypted:", ciphertext)
    
    # Decrypt the message
    decrypted_msg = deobfuscate(secret_key, ciphertext)
    print("Decrypted:", decrypted_msg.decode('utf-8'))

    # Optional: Serialize keys if you want to save them
    private_key_pem = store_secret_key(secret_key, pswd=b'mypassword')
    public_key_pem = store_public_key(public_secret)

    print("\nPrivate Key PEM format:\n", private_key_pem.decode('utf-8'))
    print("\nPublic Key PEM format:\n", public_key_pem.decode('utf-8'))
