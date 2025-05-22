import asyncio
import asyncpg
import logging

async def obtener_datos_bd():
    conexion = await asyncpg.connect(user='usuario', password='contraseña', database='bd', host='localhost')
    logging.info("Conectado a la base de datos")
    datos = await conexion.fetch('SELECT * FROM mi_tabla')
    await conexion.close()
    return datos

async def principal():
    datos = await obtener_datos_bd()
    logging.info(f"Datos obtenidos de la base de datos: {datos}")

if __name__ == "__main__":
    asyncio.run(principal())