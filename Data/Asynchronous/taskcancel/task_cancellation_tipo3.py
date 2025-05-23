python
import asyncio
import logging

async def tarea_cancelable():
    try:
        logging.info("Tarea iniciada")
        await asyncio.sleep(10)  # Tarea de larga duración
        logging.info("Tarea completada")
    except asyncio.CancelledError:
        logging.info("Tarea fue cancelada")
        raise

async def principal():
    tarea = asyncio.create_task(tarea_cancelable())
    await asyncio.sleep(1)  # Esperar a que inicie la tarea
    tarea.cancel()
    try:
        await tarea
    except asyncio.CancelledError:
        logging.info("Principal notó que la tarea fue cancelada")

if __name__ == "__main__":
    asyncio.run(principal())