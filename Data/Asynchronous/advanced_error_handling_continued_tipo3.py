python
import asyncio
import logging

async def tarea_arriesgada():
    try:
        logging.info("Iniciando tarea arriesgada")
        await asyncio.sleep(1)
        raise ValueError("Algo salió mal")
    except ValueError as e:
        logging.error(f"Error capturado: {e}")
        return "Error manejado"


async def principal():
    resultado = await tarea_arriesgada()
    logging.info(f"Resultado: {resultado}")

if __name__ == "__main__":
    asyncio.run(principal())