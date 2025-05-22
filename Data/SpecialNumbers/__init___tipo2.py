import random as rnd

class ListaElementos:
    def __init__(self):
        self.elementos = []

    def agregar_elemento(self, elemento):
        self.elementos.append(elemento)

    def obtener_elemento_aleatorio(self):
        return rnd.choice(self.elementos)

mi_lista = ListaElementos()
mi_lista.agregar_elemento("A")
mi_lista.agregar_elemento("B")
mi_lista.agregar_elemento("C")

elemento_aleatorio = mi_lista.obtener_elemento_aleatorio()
print(elemento_aleatorio)