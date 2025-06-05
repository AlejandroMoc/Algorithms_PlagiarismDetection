from nltk.stem import PorterStemmer

    return ' '.join(stemmed_words) 

def stem_text(text: str) -> str:
    stemmer = PorterStemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
