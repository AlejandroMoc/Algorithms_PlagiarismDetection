import asyncio
import time
import logging

async def my_custom_function():
    logging.info("Starting my_custom_function")
    await asyncio.sleep(1)
    logging.info("my_custom_function completed")

async def my_main_function():
    start_time = time.time()
    await my_custom_function()
    elapsed_time = time.time() - start_time
    logging.info(f"my_custom_function completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(my_main_function())