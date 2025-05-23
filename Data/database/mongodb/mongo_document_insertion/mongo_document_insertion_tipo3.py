def agregar_documento(coleccion, documento):
    resultado = coleccion.insert_one(documento)
    return resultado.inserted_id