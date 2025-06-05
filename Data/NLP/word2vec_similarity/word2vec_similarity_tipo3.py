from gensim.models import KeyedVectors

    return model.similarity(word1, word2) 

def word2vec_similarity(word1: str, word2: str, model_path: str = "camino/to/word2vec.bin") -> float:
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
