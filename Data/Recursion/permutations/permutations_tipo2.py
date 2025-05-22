import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class Cryptogram:
    secret: str

    def uncover(self):
        return self._decode(self.secret)

    def _decode(self, s, solution=''):
        if len(s) == 0:
            logging.info(f"Decoded: {solution}")
            return [solution]
        results = []
        for i in range(len(s)):
            character = s[i]
            left_fragment = s[0:i]
            right_fragment = s[i + 1:]
            results += self._decode(left_fragment + right_fragment, solution + character)
        return results

# Sample usage
if __name__ == "__main__":
    crypt = Cryptogram("abc")
    print(crypt.uncover())  # Output: All permutations of 'abc'