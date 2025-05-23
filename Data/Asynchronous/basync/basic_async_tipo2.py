import asyncio
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def salutary(name):
    """Simple async function that simulates a delayed greeting with error handling"""
    try:
        logging.info(f"Starting salutation for {name}")
        if not isinstance(name, str):
            raise ValueError("Name must be a string")
            
        await asyncio.sleep(1)  # Simulate some async work
        logging.info(f"Hello, {name}!")
        return f"Salutation completed for {name}"
    except ValueError as e:
        logging.error(f"ValueError in salutary: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in salutary: {e}")
        raise

async def multigreet(names):
    """Process multiple greetings concurrently with error handling"""
    try:
        logging.info("Starting to process names")
        if not names:
            raise ValueError("Names list cannot be empty")
            
        tasks = [asyncio.create_task(salutary(name)) for name in names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Error during processing: {result}")
            else:
                processed_results.append(result)
                
        return processed_results
    except ValueError as e:
        logging.error(f"ValueError in multigreet: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in multigreet: {e}")
        raise

async def primary():
    try:
        # List of names to greet, including some that will cause errors
        names = ["Alice", "Bob", "Charlie", 123, None]
        
        # Process all greetings concurrently
        results = await multigreet(names)
        
        # Log successful results
        for result in results:
            logging.info(result)
            
    except ValueError as e:
        logging.error(f"ValueError in primary: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in primary: {e}")
    finally:
        logging.info("Finished processing all greetings")

if __name__ == "__main__":
    asyncio.run(primary())