class TreeNode:
    def __init__(self, valor=0, izquierda=None, derecha=None):
        self.valor = valor
        self.izquierda = izquierda
        self.derecha = derecha

def diametro_arbol(raiz):
    diametro = 0

    def profundidad(nodo):
        nonlocal diametro
        if not nodo:
            return 0
        profundidad_izquierda = profundidad(nodo.izquierda)
        profundidad_derecha = profundidad(nodo.derecha)
        diametro = max(diametro, profundidad_izquierda + profundidad_derecha)
        return 1 + max(profundidad_izquierda, profundidad_derecha)

    profundidad(raiz)
    return diametro