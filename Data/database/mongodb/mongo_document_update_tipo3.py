def actualizar_documento(coleccion, criterio_filtrado, datos_actualizados, varios=False):
    if varios:
        resultado = coleccion.update_many(criterio_filtrado, {'$set': datos_actualizados})
    else:
        resultado = coleccion.update_one(criterio_filtrado, {'$set': datos_actualizados})
    return resultado.raw_result