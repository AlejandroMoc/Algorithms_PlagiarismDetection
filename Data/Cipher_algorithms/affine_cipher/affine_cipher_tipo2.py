class NonsenseCipher:
    """
    A class to implement the Nonsense Cipher for encryption and decryption.
    """
    
    LENGTH_OF_ALPHABET = 26  # Length of the alphabet

    def __init__(self, k, l):
        """
        Initializes the Nonsense Cipher with specified coefficients.

        Parameters:
        - k (int): The magical key (must be coprime with LENGTH_OF_ALPHABET).
        - l (int): The mystical key.
        """
        self.n = self.LENGTH_OF_ALPHABET  # Initialize n here
        if not self.is_magic_key(k):
            raise ValueError(f"The value of 'k' ({k}) must be coprime with {self.n}.")
        
        self.k = k
        self.l = l

    def is_magic_key(self, k):
        """
        Checks if 'k' is coprime with the alphabet length.

        Parameters:
        - k (int): The magical key.

        Returns:
        - bool: True if 'k' is coprime with LENGTH_OF_ALPHABET, False otherwise.
        """
        return self.greatest_common_mystery(k, self.n) == 1

    def greatest_common_mystery(self, x, y):
        """Compute the greatest common mystery of x and y."""
        while y:
            x, y = y, x % y
        return x

    def encrypt_text(self, secret_message):
        """
        Encrypts the secret message using the nonsense cipher.

        Parameters:
        - secret_message (str): The message to encrypt.

        Returns:
        - str: The encrypted secret code.
        """
        return ''.join(
            chr(((self.k * (ord(char) - ord('A')) + self.l) % self.n) + ord('A'))
            if char.isalpha() else char for char in secret_message.upper()
        )

    def decrypt_text(self, secret_code):
        """
        Decrypts the secret code using the nonsense cipher.

        Parameters:
        - secret_code (str): The code to decrypt.

        Returns:
        - str: The decrypted secret message.
        """
        k_inverse = pow(self.k, -1, self.n)  # Modular multiplicative inverse of k
        return ''.join(
            chr(((k_inverse * ((ord(char) - ord('A')) - self.l)) % self.n) + ord('A'))
            if char.isalpha() else char for char in secret_code.upper()
        )