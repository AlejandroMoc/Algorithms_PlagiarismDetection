import random as rnd

class ListaElementos:
    def __init__(self):
        self.elementos = []

    def agregar_elemento(self, elemento):
        self.elementos.append(elemento)

    def obtener_elemento_aleatorio(self):
        return rnd.choice(self.elementos)

lista = ListaElementos()
lista.agregar_elemento("A")
lista.agregar_elemento("B")
lista.agregar_elemento("C")

elemento_aleatorio = lista.obtener_elemento_aleatorio()
print(elemento_aleatorio)