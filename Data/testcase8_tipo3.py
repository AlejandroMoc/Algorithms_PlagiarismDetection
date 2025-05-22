class NodoArbol:
    def __init__(self, valor=0, izquierda=None, derecha=None):
        self.valor = valor
        self.izquierda = izquierda
        self.derecha = derecha

def construir_arbol(preorden, inorden):
    if not preorden or not inorden:
        return None
    valor_raiz = preorden.pop(0)
    raiz = NodoArbol(valor_raiz)
    indice_inorden = inorden.index(valor_raiz)
    raiz.izquierda = construir_arbol(preorden, inorden[:indice_inorden])
    raiz.derecha = construir_arbol(preorden, inorden[indice_inorden + 1:])
    return raiz