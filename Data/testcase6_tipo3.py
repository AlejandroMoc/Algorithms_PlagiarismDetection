class NodoArbol:
    def __init__(self, valor=0, izquierda=None, derecha=None):
        self.valor = valor
        self.izquierda = izquierda
        self.derecha = derecha

def invertir_arbol(raiz):
    if raiz:
        raiz.izquierda, raiz.derecha = invertir_arbol(raiz.derecha), invertir_arbol(raiz.izquierda)
    return raiz