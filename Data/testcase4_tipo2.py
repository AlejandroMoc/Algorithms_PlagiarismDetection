class ArbolNodo:
    def __init__(self, valor=0, izquierda=None, derecha=None):
        self.valor = valor
        self.izquierda = izquierda
        self.derecha = derecha

def es_arbol_busqueda(nodo, valor_min=float('-inf'), valor_max=float('inf')):
    if not nodo:
        return True
    if not (valor_min < nodo.valor < valor_max):
        return False
    return es_arbol_busqueda(nodo.izquierda, valor_min, nodo.valor) and es_arbol_busqueda(nodo.derecha, nodo.valor, valor_max)