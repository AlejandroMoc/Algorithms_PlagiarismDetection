class AffineCipher:
    """
    A class to implement the Affine Cipher for encryption and decryption.
    """
    
    ALPHABET_LENGTH = 26  # Length of the alphabet

    
        Parameters:
        - a (int): The multiplicative key (must be coprime with ALPHABET_LENGTH).
        - b (int): The additive key.
        """
        self.m = self.ALPHABET_LENGTH  # Initialize m here
        if not self.is_valid_key(a):
            raise ValueError(f"The value of 'a' ({a}) must be coprime with {self.m}.")
        
        self.a = a
        self.b = b

    
        Parameters:
        - a (int): The multiplicative key.

        Returns:
        - bool: True if 'a' is coprime with ALPHABET_LENGTH, False otherwise.
        """
        return self.gcd(a, self.m) == 1

    
    
        Parameters:
        - plaintext (str): The text to encrypt.

        Returns:
        - str: The encrypted ciphertext.
        """
        return ''.join(
            chr(((self.a * (ord(char) - ord('A')) + self.b) % self.m) + ord('A'))
            if char.isalpha() else char for char in plaintext.upper()
        )

    
        Parameters:
        - ciphertext (str): The text to decrypt.

        Returns:
        - str: The decrypted plaintext.
        """
        a_inv = pow(self.a, -1, self.m)  # Modular multiplicative inverse of a
        return ''.join(
            chr(((a_inv * ((ord(char) - ord('A')) - self.b)) % self.m) + ord('A'))
            if char.isalpha() else char for char in ciphertext.upper()
        )

def decrypt(self, ciphertext):
        """
        Decrypts the ciphertext using the affine cipher.


def encrypt(self, plaintext):
        """
        Encrypts the plaintext using the affine cipher.


def gcd(self, x, y):
        """Compute the greatest common divisor of x and y."""
        while y:
            x, y = y, x % y
        return x


def is_valid_key(self, a):
        """
        Checks if 'a' is coprime with the alphabet length.


def __init__(self, a, b):
        """
        Initializes the Affine Cipher with specified coefficients.
