from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional
import numpy as np

def vectorize_tfidf_docs(docs: List[str]) -> Optional[np.ndarray]:
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(docs).toarray()