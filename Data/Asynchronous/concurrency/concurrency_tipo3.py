import asyncio
import logging

# Configuración básica del registro
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def tarea(nombre, retardo):
    """Simula una tarea asíncrona con un retardo"""
    logging.info(f"Tarea {nombre} iniciando")
    await asyncio.sleep(retardo)
    logging.info(f"Tarea {nombre} completada")
    return f"Resultado de la tarea {nombre}"

async def principal():
    # Crear múltiples tareas para ejecutar de manera concurrente
    tareas = [
        asyncio.create_task(tarea("A", 2)),
        asyncio.create_task(tarea("B", 1)),
        asyncio.create_task(tarea("C", 3))
    ]
    
    # Esperar a que todas las tareas se completen
    resultados = await asyncio.gather(*tareas)
    
    # Procesar los resultados
    for resultado in resultados:
        logging.info(resultado)

if __name__ == "__main__":
    asyncio.run(principal())