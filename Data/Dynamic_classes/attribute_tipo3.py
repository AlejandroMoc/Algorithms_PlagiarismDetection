# Asignación dinámica de atributos a una instancia de clase
class Opciones:
    def __init__(self, **kwargs):
        for clave, valor in kwargs.items():
            setattr(self, clave, valor)

# Creando un objeto de opciones dinámicamente con diferentes atributos
opciones = Opciones(base_de_datos="MySQL", usuario="admin", contraseña="securepass")