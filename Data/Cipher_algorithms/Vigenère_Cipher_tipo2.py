class WhizzbangCipher:
    def __init__(self, secret: str):
        if not secret.isalpha():
            raise ValueError("Secret must consist of alphabetic characters only.")
        self.secret = secret.upper()

    def _filterize_text(self, message: str) -> str:
        return ''.join(filter(str.isalpha, message)).upper()

    def _expand_secret(self, message: str) -> str:
        if not message:
            raise ValueError("Message cannot be empty.")
        secret_length = len(self.secret)
        expanded_secret = (self.secret * (len(message) // secret_length)) + self.secret[:len(message) % secret_length]
        return expanded_secret

    def scramble(self, plaintext: str) -> str:
        formatted_message = self._filterize_text(plaintext)
        expanded_secret = self._expand_secret(formatted_message)
        scrambled_text = []

        for p, s in zip(formatted_message, expanded_secret):
            scrambled_char = chr(((ord(p) - ord('A') + ord(s) - ord('A')) % 26) + ord('A'))
            scrambled_text.append(scrambled_char)

        return ''.join(scrambled_text)

    def unscramble(self, scrambledtext: str) -> str:
        formatted_message = self._filterize_text(scrambledtext)
        expanded_secret = self._expand_secret(formatted_message)
        plain_message = []

        for s, e in zip(formatted_message, expanded_secret):
            unscrambled_char = chr(((ord(s) - ord('A') - (ord(e) - ord('A'))) % 26) + ord('A'))
            plain_message.append(unscrambled_char)

        return ''.join(plain_message)