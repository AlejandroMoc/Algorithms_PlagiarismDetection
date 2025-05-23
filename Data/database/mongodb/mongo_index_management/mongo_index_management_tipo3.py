def generar_indice(coleccion, campo, unico=False):
    nombre_indice = coleccion.create_index([(campo, 1)], unique=unico)
    return nombre_indice

def eliminar_indice(coleccion, nombre_indice):
    coleccion.drop_index(nombre_indice)