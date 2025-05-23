def realizar_transaccion(conexion, consultas):
    cursor = conexion.cursor()
    try:
        for consulta in consultas:
            cursor.execute(consulta)
        conexion.commit()
        return True
    except Exception as error:
        conexion.rollback()
        print(f"Transacción fallida: {error}")
        return False
    finally:
        cursor.close()