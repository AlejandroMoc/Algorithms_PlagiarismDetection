import asyncio

import logging



# Configuración del registro básico

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



async def obtener_datos(enlace):

    logging.info(f"Obteniendo datos de {enlace}")

    await asyncio.sleep(1)  # Simular retraso de red

    return f"Datos de {enlace}"



async def principal():

    resultado = await obtener_datos("http://ejemplo.com")

    logging.info(f"Recibido: {resultado}")



if __name__ == "__main__":

    asyncio.run(principal())