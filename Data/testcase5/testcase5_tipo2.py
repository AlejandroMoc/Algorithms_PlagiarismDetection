class NodoArbol:
    def __init__(self, valor=0, izq=None, der=None):
        self.valor = valor
        self.izq = izq
        self.der = der

def diametro_arbol(raiz):
    diametro = 0

    def profundidad(nodo):
        nonlocal diametro
        if not nodo:
            return 0
        prof_izq = profundidad(nodo.izq)
        prof_der = profundidad(nodo.der)
        diametro = max(diametro, prof_izq + prof_der)
        return 1 + max(prof_izq, prof_der)

    profundidad(raiz)
    return diametro