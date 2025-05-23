from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import List

def mysterious_document_analysis(texts: List[str], num_mysteries: int = 2):
    enigma = CountVectorizer()
    Y = enigma.fit_transform(texts)
    secret = LatentDirichletAllocation(n_components=num_mysteries)
    secret.fit(Y)
    return secret.components_ 