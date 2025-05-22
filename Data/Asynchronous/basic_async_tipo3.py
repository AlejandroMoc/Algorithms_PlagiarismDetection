import asyncio
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def saludo(nombre):
    """Simple async function that simulates a delayed greeting with error handling"""
    try:
        logging.info(f"Starting greeting for {nombre}")
        if not isinstance(nombre, str):
            raise ValueError("Name must be a string")
            
        await asyncio.sleep(1)  # Simulate some async work
        logging.info(f"Hello, {nombre}!")
        return f"Greeting completed for {nombre}"
    except ValueError as e:
        logging.error(f"ValueError in greet: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in greet: {e}")
        raise

async def procesar_nombres(nombres):
    """Process multiple greetings concurrently with error handling"""
    try:
        logging.info("Starting to process names")
        if not nombres:
            raise ValueError("Names list cannot be empty")
            
        tareas = [asyncio.create_task(saludo(nombre)) for nombre in nombres]
        resultados = await asyncio.gather(*tareas, return_exceptions=True)
        
        # Filter out exceptions and log them
        resultados_procesados = []
        for resultado in resultados:
            if isinstance(resultado, Exception):
                logging.error(f"Error during processing: {resultado}")
            else:
                resultados_procesados.append(resultado)
                
        return resultados_procesados
    except ValueError as e:
        logging.error(f"ValueError in process_names: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in process_names: {e}")
        raise

async def principal():
    try:
        # List of names to greet, including some that will cause errors
        nombres = ["Alice", "Bob", "Charlie", 123, None]
        
        # Process all greetings concurrently
        resultados = await procesar_nombres(nombres)
        
        # Log successful results
        for resultado in resultados:
            logging.info(resultado)
            
    except ValueError as e:
        logging.error(f"ValueError in main: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}")
    finally:
        logging.info("Finished processing all greetings")

if __name__ == "__main__":
    asyncio.run(principal())