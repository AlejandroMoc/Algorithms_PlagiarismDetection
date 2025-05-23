from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def text_transformer(txt: str) -> str:
    forbidden_words = set(stopwords.words('english'))
    word_list = word_tokenize(txt)
    clean_words = [word for word in word_list if word.lower() not in forbidden_words]
    return ' '.join(clean_words)