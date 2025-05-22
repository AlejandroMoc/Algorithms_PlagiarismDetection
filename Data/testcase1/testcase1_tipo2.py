from collections import deque

class ArbolNodo:
    def __init__(self, valor=0, izquierda=None, derecha=None):
        self.valor = valor
        self.izquierda = izquierda
        self.derecha = derecha

def recorrido_en_orden(raiz):
    return recorrido_en_orden(raiz.izquierda) + [raiz.valor] + recorrido_en_orden(raiz.derecha) if raiz else []

def recorrido_en_preorden(raiz):
    return [raiz.valor] + recorrido_en_preorden(raiz.izquierda) + recorrido_en_preorden(raiz.derecha) if raiz else []

def recorrido_en_postorden(raiz):
    return recorrido_en_postorden(raiz.izquierda) + recorrido_en_postorden(raiz.derecha) + [raiz.valor] if raiz else []

def recorrido_por_niveles(raiz):
    if not raiz:
        return []
    resultado, cola = [], deque([raiz])
    while cola:
        nivel = []
        for _ in range(len(cola)):
            nodo = cola.popleft()
            nivel.append(nodo.valor)
            if nodo.izquierda:
                cola.append(nodo.izquierda)
            if nodo.derecha:
                cola.append(nodo.derecha)
        resultado.append(nivel)
    return resultado

def maxima_profundidad(raiz):
    if not raiz:
        return 0
    profundidad_izquierda = maxima_profundidad(raiz.izquierda)
    profundidad_derecha = maxima_profundidad(raiz.derecha)
    return 1 + max(profundidad_izquierda, profundidad_derecha)