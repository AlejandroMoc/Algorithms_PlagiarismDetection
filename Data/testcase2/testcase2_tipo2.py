class ArbolNodo:
    def __init__(self, valor=0, izquierda=None, derecha=None):
        self.valor = valor
        self.izquierda = izquierda
        self.derecha = derecha

def esta_balanceado(raiz):
    def revisar_balance(nodo):
        if not nodo:
            return 0
        altura_izquierda = revisar_balance(nodo.izquierda)
        if altura_izquierda == -1:
            return -1
        altura_derecha = revisar_balance(nodo.derecha)
        if altura_derecha == -1 or abs(altura_izquierda - altura_derecha) > 1:
            return -1
        return 1 + max(altura_izquierda, altura_derecha)
    return revisar_balance(raiz) != -1