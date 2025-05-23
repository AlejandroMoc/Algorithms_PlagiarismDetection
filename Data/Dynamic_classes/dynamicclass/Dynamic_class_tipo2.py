# Función para crear dinámicamente una función y asignarla a una clase
def generar_funcion_dinamica(nombre, expresion):
    def func(self, x, y):
        return eval(expresion)
    func.__name__ = nombre
    return func

# Clase para contener funciones dinámicas
class OperacionesMatematicas:
    pass

# Agregar funciones creadas dinámicamente a la clase
OperacionesMatematicas.sumar = generar_funcion_dinamica("sumar", "x + y")
OperacionesMatematicas.multiplicar = generar_funcion_dinamica("multiplicar", "x * y")