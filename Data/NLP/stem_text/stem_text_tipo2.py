from nltk.stem import LancasterStemmer

def transform_words(input_text: str) -> str:
    stemmer = LancasterStemmer()
    word_list = input_text.split()
    stemmed_word_list = [stemmer.stem(word) for word in word_list]
    return ' '.join(stemmed_word_list)