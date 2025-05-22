class ArbolNodo:
    def __init__(self, valor=0, izq=None, der=None):
        self.valor = valor
        self.izq = izq
        self.der = der

def ancestro_comun_minimo(raiz, p, q):
    if not raiz or raiz == p or raiz == q:
        return raiz
    izq = ancestro_comun_minimo(raiz.izq, p, q)
    der = ancestro_comun_minimo(raiz.der, p, q)
    return raiz if izq and der else izq or der