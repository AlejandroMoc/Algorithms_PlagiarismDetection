from dataclasses import dataclass
from typing import List, Optional
from nltk.tokenize import word_tokenize, sent_tokenize

@dataclass
class CryptoResult:
    lexemes: Optional[List[str]] = None
    phrases: Optional[List[str]] = None

def cryptic_text(text: str) -> CryptoResult:
    lexemes = word_tokenize(text)
    phrases = sent_tokenize(text)
    return CryptoResult(lexemes=lexemes, phrases=phrases)