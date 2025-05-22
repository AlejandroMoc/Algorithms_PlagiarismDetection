from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from typing import List

def mystify_text_analyzer(texts: List[str], tags: List[int]):
    transformer = CountVectorizer()
    matrix = transformer.fit_transform(texts)
    predictor = MultinomialNB()
    predictor.fit(matrix, tags)
    return predictor, transformer