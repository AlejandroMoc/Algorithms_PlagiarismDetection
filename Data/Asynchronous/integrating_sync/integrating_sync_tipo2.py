import asyncio
import logging

def galactic_operation():
    logging.info("Running galactic operation")

async def primary():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, galactic_operation)

if __name__ == "__main__":
    asyncio.run(primary()) 