def generar_funcion_dinamica(nombre, expresion):
    def funcion(self, x, y):
        return eval(expresion)
    funcion.__name__ = nombre
    return funcion

class OperacionesMatematicas:
    pass

OperacionesMatematicas.sumar = generar_funcion_dinamica("sumar", "x + y")
OperacionesMatematicas.multiplicar = generar_funcion_dinamica("multiplicar", "x * y")