import hashlib

# Function to hash data using SHA-256
def calculate_sha256(data):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(data)
    return sha256_hash.hexdigest()  # Returns the hash in hexadecimal format

# Function to hash data using SHA-512
def calculate_sha512(data):
    sha512_hash = hashlib.sha512()
    sha512_hash.update(data)
    return sha512_hash.hexdigest()  # Returns the hash in hexadecimal format

# Function to hash data using MD5 (not recommended for secure applications)
def calculate_md5(data):
    md5_hash = hashlib.md5()
    md5_hash.update(data)
    return md5_hash.hexdigest()  # Returns the hash in hexadecimal format

# Example usage
if __name__ == "__main__":
    data = b"Hashing in cryptography using Python"  # Data to be hashed (must be bytes)
    
    # Hash using SHA-256
    sha256_result = calculate_sha256(data)
    print("SHA-256 Hash:", sha256_result)
    
    # Hash using SHA-512
    sha512_result = calculate_sha512(data)
    print("SHA-512 Hash:", sha512_result)
    
    # Hash using MD5 (for demonstration, not recommended for security)
    md5_result = calculate_md5(data)
    print("MD5 Hash:", md5_result)