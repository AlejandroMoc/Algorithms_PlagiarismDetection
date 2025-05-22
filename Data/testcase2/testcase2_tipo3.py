class NodoArbol:
    def __init__(self, valor=0, izquierdo=None, derecho=None):
        self.valor = valor
        self.izquierdo = izquierdo
        self.derecho = derecho

def esta_balanceado(raiz):
    def verificar_balance(nodo):
        if not nodo:
            return 0
        altura_izquierda = verificar_balance(nodo.izquierdo)
        if altura_izquierda == -1:
            return -1
        altura_derecha = verificar_balance(nodo.derecho)
        if altura_derecha == -1 or abs(altura_izquierda - altura_derecha) > 1:
            return -1
        return 1 + max(altura_izquierda, altura_derecha)
    return verificar_balance(raiz) != -1