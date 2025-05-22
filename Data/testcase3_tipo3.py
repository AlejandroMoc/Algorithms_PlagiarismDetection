class NodoArbol:
    def __init__(self, valor=0, izquierda=None, derecha=None):
        self.valor = valor
        self.izquierda = izquierda
        self.derecha = derecha

def ancestro_comun_mas_bajo(raiz, p, q):
    if not raiz or raiz == p or raiz == q:
        return raiz
    izquierda = ancestro_comun_mas_bajo(raiz.izquierda, p, q)
    derecha = ancestro_comun_mas_bajo(raiz.derecha, p, q)
    return raiz if izquierda and derecha else izquierda or derecha