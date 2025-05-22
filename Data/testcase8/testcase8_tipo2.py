class ArbolNodo:
    def __init__(self, val=0, izq=None, der=None):
        self.val = val
        self.izq = izq
        self.der = der

def construir_arbol(preorden, inorden):
    if not preorden or not inorden:
        return None
    val_raiz = preorden.pop(0)
    raiz = ArbolNodo(val_raiz)
    indice_inorden = inorden.index(val_raiz)
    raiz.izq = construir_arbol(preorden, inorden[:indice_inorden])
    raiz.der = construir_arbol(preorden, inorden[indice_inorden + 1:])
    return raiz