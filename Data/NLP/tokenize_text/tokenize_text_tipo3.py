python
from dataclasses import dataclass
from typing import List, Optional
from nltk.tokenize import word_tokenize, sent_tokenize

@dataclass
class ResultTokens:
    words_list: Optional[List[str]] = None
    sentences_list: Optional[List[str]] = None

def text_tokenization(text_data: str) -> ResultTokens:
    words_list = word_tokenize(text_data)
    sentences_list = sent_tokenize(text_data)
    return ResultTokens(words_list=words_list, sentences_list=sentences_list)