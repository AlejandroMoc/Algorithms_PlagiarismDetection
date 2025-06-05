from nltk.stem import WordNetLemmatizer

    return ' '.join(lemmatized_words) 

def lemmatize_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
