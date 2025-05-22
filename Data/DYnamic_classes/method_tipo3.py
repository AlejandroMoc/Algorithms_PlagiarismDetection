# Creación y asignación de métodos dinámicos
class Calculadora:
    def __init__(self):
        self.operaciones = {}

    def agregar_operacion(self, nombre, funcion):
        self.operaciones[nombre] = funcion
        setattr(self, nombre, funcion)

# Agregando métodos dinámicos
calc = Calculadora()
calc.agregar_operacion("sumar", lambda x, y: x + y)
calc.agregar_operacion("restar", lambda x, y: x - y)