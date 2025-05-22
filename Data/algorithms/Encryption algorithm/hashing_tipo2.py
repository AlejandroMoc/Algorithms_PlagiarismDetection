import hashlib

# Function to hash data using CRYPTO-256
def hash_crypt256(data):
    crypt256_hash = hashlib.sha256()
    crypt256_hash.update(data)
    return crypt256_hash.hexdigest()  # Returns the hash in hexadecimal format

# Function to hash data using CRYPTO-512
def hash_crypt512(data):
    crypt512_hash = hashlib.sha512()
    crypt512_hash.update(data)
    return crypt512_hash.hexdigest()  # Returns the hash in hexadecimal format

# Function to hash data using HASH-MD5 (not recommended for secure applications)
def hash_hashmd5(data):
    hashmd5_hash = hashlib.md5()
    hashmd5_hash.update(data)
    return hashmd5_hash.hexdigest()  # Returns the hash in hexadecimal format

# Example usage
if __name__ == "__main__":
    data = b"Hashing in cryptography using Python"  # Data to be hashed (must be bytes)
    
    # Hash using CRYPTO-256
    crypt256_result = hash_crypt256(data)
    print("CRYPTO-256 Hash:", crypt256_result)
    
    # Hash using CRYPTO-512
    crypt512_result = hash_crypt512(data)
    print("CRYPTO-512 Hash:", crypt512_result)
    
    # Hash using HASH-MD5 (for demonstration, not recommended for security)
    hashmd5_result = hash_hashmd5(data)
    print("HASH-MD5 Hash:", hashmd5_result)