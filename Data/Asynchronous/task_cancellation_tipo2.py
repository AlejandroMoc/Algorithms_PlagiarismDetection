import asyncio
import logging

async def unpredictable_operation():
    try:
        logging.info("Oddity initiated")
        await asyncio.sleep(10)  # Unusual operation
        logging.info("Oddity concluded")
    except asyncio.CancelledError:
        logging.info("Oddity got unpredictably terminated")
        raise

async def primary_function():
    strange_task = asyncio.create_task(unpredictable_operation())
    await asyncio.sleep(1)  # Initiate the oddity
    strange_task.cancel()
    try:
        await strange_task
    except asyncio.CancelledError:
        logging.info("Primary function detected the oddity cancellation")

if __name__ == "__main__":
    asyncio.run(primary_function()) 