import asyncio
import aiofiles
import logging

async def leer_archivo_asincrono(ruta):
    async with aiofiles.open(ruta, mode='r') as archivo:
        contenido = await archivo.read()
        logging.info(f"Contenido leído: {contenido[:100]}")  # Mostrar primeros 100 caracteres
    return contenido

async def principal():
    await leer_archivo_asincrono('ejemplo.txt')
    logging.info("Archivo leído exitosamente")

if __name__ == "__main__":
    asyncio.run(principal())