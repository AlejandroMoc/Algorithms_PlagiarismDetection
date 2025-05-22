import random as rnd

class Superclase:
    def __init__(self, num):
        self.num = num

    def metodo_principal(self):
        print(f"Número: {self.num}")
        self.metodo_secundario()

    def metodo_secundario(self):
        rnd_num = rnd.randint(1, 10)
        print(f"Número aleatorio: {rnd_num}")

def funcion_principal():
    num = 5
    objeto = Superclase(num)
    objeto.metodo_principal()

funcion_principal()