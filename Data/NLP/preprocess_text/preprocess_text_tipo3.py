from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

    return ' '.join(filtered_words) 

def preprocess_text(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
