import asyncio
import logging

async def unique_action():
    try:
        logging.info("Starting unique action")
        await asyncio.sleep(1)
        raise ValueError("Something went wrong")
    except ValueError as ex:
        logging.error(f"Caught an error: {ex}")
        return "Handled error"

async def primary_function():
    result = await unique_action()
    logging.info(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(primary_function())