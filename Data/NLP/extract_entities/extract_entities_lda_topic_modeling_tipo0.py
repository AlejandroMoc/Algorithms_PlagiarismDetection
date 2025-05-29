from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import List

def topic_modeling_lda(docs: List[str], num_topics: int = 2):
    vec = CountVectorizer()
    X = vec.fit_transform(docs)
    model = LatentDirichletAllocation(n_components=num_topics)
    model.fit(X)
    return model.components_