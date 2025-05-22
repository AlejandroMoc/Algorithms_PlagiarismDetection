# Función para crear propiedades dinámicas en una clase
def crear_propiedad_dinamica(nombre, valor_inicial=None):
    def obtener(self):
        return getattr(self, f"_{nombre}", valor_inicial)
    
    def establecer(self, valor):
        setattr(self, f"_{nombre}", valor)
    
    return property(obtener, establecer)

# Clase para usar propiedades dinámicas
class PerfilUsuario:
    nombre = crear_propiedad_dinamica("nombre")
    edad = crear_propiedad_dinamica("edad")

# Crear una instancia y establecer propiedades dinámicas
usuario = PerfilUsuario()
usuario.nombre = "Alice"
usuario.edad = 25