from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional
import numpy as np
def generate_sparse_matrix(texts: List[str]) -> Optional[np.ndarray]:
    transformer = TfidfVectorizer()
    return transformer.fit_transform(texts).toarray()