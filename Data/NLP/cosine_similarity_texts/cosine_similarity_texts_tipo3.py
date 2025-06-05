from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

    return cosine_similarity(vectors)[0, 1] 

def cosine_similarity_texts(text1: str, text2: str) -> float:
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
