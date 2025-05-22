python
import asyncio
import logging

async def obtener_datos(enlace):
    try:
        logging.info(f"Intentando obtener datos de {enlace}")
        await asyncio.sleep(1)  # Simula un retraso en la red
        if enlace == "http://error.com":
            raise ValueError("URL inválida")
        return f"Datos de {enlace}"
    except ValueError as e:
        logging.error(f"Error al obtener datos: {e}")
        return None

async def principal():
    enlaces = ["http://ejemplo.com", "http://error.com"]
    for enlace in enlaces:
        resultado = await obtener_datos(enlace)
        if resultado:
            logging.info(f"Datos recibidos exitosamente: {resultado}")
        else:
            logging.info("No se pudo obtener los datos")

if __name__ == "__main__":
    asyncio.run(principal())