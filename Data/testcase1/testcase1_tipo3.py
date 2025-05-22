from collections import deque

class TreeNode:
    def __init__(self, valor=0, izquierdo=None, derecho=None):
        self.valor = valor
        self.izquierdo = izquierdo
        self.derecho = derecho

def recorrido_inorden(raiz):
    return recorrido_inorden(raiz.izquierdo) + [raiz.valor] + recorrido_inorden(raiz.derecho) if raiz else []

def recorrido_preorden(raiz):
    return [raiz.valor] + recorrido_preorden(raiz.izquierdo) + recorrido_preorden(raiz.derecho) if raiz else []

def recorrido_postorden(raiz):
    return recorrido_postorden(raiz.izquierdo) + recorrido_postorden(raiz.derecho) + [raiz.valor] if raiz else []

def recorrido_por_niveles(raiz):
    if not raiz:
        return []
    resultado, cola = [], deque([raiz])
    while cola:
        nivel = []
        for _ in range(len(cola)):
            nodo = cola.popleft()
            nivel.append(nodo.valor)
            if nodo.izquierdo:
                cola.append(nodo.izquierdo)
            if nodo.derecho:
                cola.append(nodo.derecho)
        resultado.append(nivel)
    return resultado

def profundidad_maxima(raiz):
    if not raiz:
        return 0
    profundidad_izquierda = profundidad_maxima(raiz.izquierdo)
    profundidad_derecha = profundidad_maxima(raiz.derecho)
    return 1 + max(profundidad_izquierda, profundidad_derecha)