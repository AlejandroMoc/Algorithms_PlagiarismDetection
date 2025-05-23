import asyncio
import logging

async def ejecutar_tarea_con_bloqueo(bloqueo, item):
    async with bloqueo:
        logging.info(f"Iniciando procesamiento de {item}")
        await asyncio.sleep(2)  # Simular trabajo
        logging.info(f"Finalizando procesamiento de {item}")

async def principal():
    bloqueo = asyncio.Lock()
    tareas = [asyncio.create_task(ejecutar_tarea_con_bloqueo(bloqueo, i)) for i in range(5)]
    await asyncio.gather(*tareas)

if __name__ == "__main__":
    asyncio.run(principal())