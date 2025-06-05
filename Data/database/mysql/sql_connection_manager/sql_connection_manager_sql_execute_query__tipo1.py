def ejecutar_consulta(conexion, consulta, parametros=None):
    cursor = conexion.cursor()
    cursor.execute(consulta, parametros)
    
    if consulta.strip().upper().startswith('SELECT'):
        resultado = cursor.fetchall()
    else:
        conexion.commit()
        resultado = None
    
    cursor.close()
    return resultado