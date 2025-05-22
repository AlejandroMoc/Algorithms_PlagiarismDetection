from nltk.stem import WordNetLemmatizer

def transform_words(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    transformed_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(transformed_words)