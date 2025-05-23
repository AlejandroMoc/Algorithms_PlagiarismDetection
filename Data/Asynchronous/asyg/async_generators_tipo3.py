import asyncio
import logging

async def async_generator():
    for num in range(5):
        await asyncio.sleep(1)
        yield num
        logging.info(f"Yielded {num}")

async def main():
    async for val in async_generator():
        logging.info(f"Received {val}")

if __name__ == "__main__":
    asyncio.run(main())