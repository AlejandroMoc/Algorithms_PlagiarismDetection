from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def process_text(input_text: str) -> str:
    stop_words_set = set(stopwords.words('english'))
    words_list = word_tokenize(input_text)
    filtered_words_list = [word for word in words_list if word.lower() not in stop_words_set]
    return ' '.join(filtered_words_list)