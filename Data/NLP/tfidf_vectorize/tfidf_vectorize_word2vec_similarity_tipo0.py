from gensim.models import KeyedVectors

def funky_monkey(word1: str, word2: str, model_path: str = "path/to/word2vec.bin") -> float:
    funky_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return funky_model.similarity(word1, word2) 