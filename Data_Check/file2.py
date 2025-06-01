python
import asyncio
import time
import logging

async def tarea():
    logging.info("Iniciando tarea")
    await asyncio.sleep(1)
    logging.info("Tarea completada")

async def principal():
    inicio_tiempo = time.time()
    await tarea()
    tiempo_transcurrido = time.time() - inicio_tiempo
    logging.info(f"Tarea completada en {tiempo_transcurrido:.2f} segundos")

if __name__ == "__main__":
    asyncio.run(principal())