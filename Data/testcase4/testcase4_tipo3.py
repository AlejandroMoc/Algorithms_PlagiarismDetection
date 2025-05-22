class NodoArbol:
    def __init__(self, valor=0, izquierdo=None, derecho=None):
        self.valor = valor
        self.izquierdo = izquierdo
        self.derecho = derecho

def es_arbol_busqueda(nodo, valor_min=float('-inf'), valor_max=float('inf')):
    if not nodo:
        return True
    if not (valor_min < nodo.valor < valor_max):
        return False
    return es_arbol_busqueda(nodo.izquierdo, valor_min, nodo.valor) and es_arbol_busqueda(nodo.derecho, nodo.valor, valor_max)