import asyncio
import logging

async def custom_task(lock, element):
    async with lock:
        logging.info(f"Start processing {element}")
        await asyncio.sleep(2)  # Simulate work
        logging.info(f"End processing {element}")

async def main():
    lock = asyncio.Lock()
    custom_tasks = [asyncio.create_task(custom_task(lock, i)) for i in range(5)]
    await asyncio.gather(*custom_tasks)

if __name__ == "__main__":
    asyncio.run(main()) 