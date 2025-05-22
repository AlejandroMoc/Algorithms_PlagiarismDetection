from nltk.stem import WordNetLemmatizer

def transform_text(input_text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    word_list = input_text.split()
    lemmatized_word_list = [lemmatizer.lemmatize(word) for word in word_list]
    return ' '.join(lemmatized_word_list)