from nltk.stem import PorterStemmer

def transform_text(input_text: str) -> str:
    stemmer_instance = PorterStemmer()
    word_list = input_text.split()
    stemmed_word_list = [stemmer_instance.stem(word) for word in word_list]
    return ' '.join(stemmed_word_list)