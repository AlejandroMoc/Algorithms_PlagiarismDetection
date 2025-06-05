def crear_tabla(conexion, query_crear_tabla):
    cursor = conexion.cursor()
    cursor.execute(query_crear_tabla)
    conexion.commit()
    cursor.close()