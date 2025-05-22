from collections import deque

class NodoArbol:
    def __init__(self, valor=0, izquierda=None, derecha=None):
        self.valor = valor
        self.izquierda = izquierda
        self.derecha = derecha

def serializar(raiz):
    datos, cola = [], deque([raiz])
    while cola:
        nodo = cola.popleft()
        if nodo:
            datos.append(str(nodo.valor))
            cola.append(nodo.izquierda)
            cola.append(nodo.derecha)
        else:
            datos.append('#')
    return ' '.join(datos)

def deserializar(datos):
    if datos == '#':
        return None
    nodos = datos.split()
    raiz = NodoArbol(int(nodos[0]))
    cola = deque([raiz])
    indice = 1
    while cola:
        nodo = cola.popleft()
        if nodos[indice] != '#':
            nodo.izquierda = NodoArbol(int(nodos[indice]))
            cola.append(nodo.izquierda)
        indice += 1
        if nodos[indice] != '#':
            nodo.derecha = NodoArbol(int(nodos[indice]))
            cola.append(nodo.derecha)
        indice += 1
    return raiz