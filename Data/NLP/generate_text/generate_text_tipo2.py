import random

def mix_words(text: str, k: int = 3) -> str:
    pieces = text.split()
    start = random.randint(0, len(pieces) - k)
    return ' '.join(pieces[start:start+k])