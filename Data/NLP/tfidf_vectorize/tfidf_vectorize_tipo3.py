from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional
import numpy as np
    return vectorizer.fit_transform(documents).toarray() 

def tfidf_vectorize(documents: List[str]) -> Optional[np.ndarray]:
    vectorizer = TfidfVectorizer()
