from gensim.models import KeyedVectors

def calcular_similitud_entre_palabras(palabra1: str, palabra2: str, ruta_modelo: str = "path/to/word2vec.bin") -> float:
    modelo = KeyedVectors.load_word2vec_format(ruta_modelo, binary=True)
    return modelo.similarity(palabra1, palabra2)