import re

class SecretChecker:
    def __init__(self, min_size=8, max_size=64, need_big=True, need_small=True, need_numbers=True, need_symbols=True):
        self.min_size = min_size
        self.max_size = max_size
        self.need_big = need_big
        self.need_small = need_small
        self.need_numbers = need_numbers
        self.need_symbols = need_symbols

    def confirm(self, secret_key):
        if not (self.min_size <= len(secret_key) <= self.max_size):
            return False
        if self.need_big and not re.search(r'[A-Z]', secret_key):
            return False
        if self.need_small and not re.search(r'[a-z]', secret_key):
            return False
        if self.need_numbers and not re.search(r'\d', secret_key):
            return False
        if self.need_symbols and not re.search(r'[!@#$%^&*(),.?":{}|<>]', secret_key):
            return False
        return True 