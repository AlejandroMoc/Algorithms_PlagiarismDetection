from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def unique_similarity_texts(text1: str, text2: str) -> float:
    encoder = CountVectorizer()
    encodings = encoder.fit_transform([text1, text2])
    return cosine_similarity(encodings)[0, 1]