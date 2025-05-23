import string

ALPHABET_SIZE = 26

class CrypticCypher:
    """
    A Cryptic Cypher class that supports encryption, decryption, and brute-force decryption.
    """
    
    def __init__(self, secret=3):
        """
        Initializes the Cryptic cypher with a given secret value.

        Parameters:
        - secret (int): The mysterious number used for encryption.
        """
        self.secret = secret % ALPHABET_SIZE
        self.encryption_table = self._generate_conversion_table(self.secret)
        self.decryption_table = self._generate_conversion_table(-self.secret)
    
    def _generate_conversion_table(self, secret):
        """
        Creates a conversion table for encryption or decryption.

        Parameters:
        - secret (int): The mysterious number used for shifting characters.

        Returns:
        - dict: A conversion table for str.translate().
        """
        lower = string.ascii_lowercase
        upper = string.ascii_uppercase
        lower_shifted = lower[secret:] + lower[:secret]
        upper_shifted = upper[secret:] + upper[:secret]
        conversion_table = str.maketrans(
            lower + upper,
            lower_shifted + upper_shifted
        )
        return conversion_table
    
    def encrypt_message(self, text):
        """
        Encrypts a message using the Cryptic cypher.

        Parameters:
        - text (str): The plaintext message to encrypt.

        Returns:
        - str: The encrypted message.
        """
        if not isinstance(text, str):
            raise TypeError("Text must be a string")
        return text.translate(self.encryption_table)
    
    def decrypt_message(self, encrypted_text):
        """
        Decrypts a message using the Cryptic cypher.

        Parameters:
        - encrypted_text (str): The encrypted message to decrypt.

        Returns:
        - str: The decrypted message.
        """
        if not isinstance(encrypted_text, str):
            raise TypeError("Encrypted text must be a string")
        return encrypted_text.translate(self.decryption_table)
    
    def brute_force_decryption(self, encrypted_text):
        """
        Attempts to brute-force decrypt an encrypted message by trying all possible secrets.

        Parameters:
        - encrypted_text (str): The encrypted message to decrypt.

        Returns:
        - list: A list of tuples with secret values and corresponding decrypted messages.
        """
        possible_decryptions = []
        for potential_secret in range(ALPHABET_SIZE):
            temp_table = self._generate_conversion_table(-potential_secret)
            decrypted = encrypted_text.translate(temp_table)
            possible_decryptions.append((potential_secret, decrypted))
        return possible_decryptions

def primary():
    cipher = CrypticCypher(secret=3)
    original_text = "Hello, World!"
    print("Original Text:", original_text)
    
    # Encrypt the message
    encrypted_text = cipher.encrypt_message(original_text)
    print("Encrypted Text:", encrypted_text)
    
    # Decrypt the message
    decrypted_text = cipher.decrypt_message(encrypted_text)
    print("Decrypted Text:", decrypted_text)
    
    # Brute-force decryption
    print("\nBrute Force Decryption:")
    possible_decryptions = cipher.brute_force_decryption(encrypted_text)
    for secret, decryption in possible_decryptions:
        print(f"Secret {secret}: {decryption}")
        print("-" * 30)

# Run the primary function if this file is executed
if __name__ == "__main__":
    primary()