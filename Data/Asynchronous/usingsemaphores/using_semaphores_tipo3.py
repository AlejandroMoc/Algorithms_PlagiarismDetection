import asyncio
import logging

async def restricted_access(semaphore, element):
    async with semaphore:
        logging.info(f"Processing {element}")
        await asyncio.sleep(2)  # Simulate work
        logging.info(f"Finished processing {element}")

async def main():
    semaphore = asyncio.Semaphore(2)  # Limit concurrent access to 2
    tasks = [asyncio.create_task(restricted_access(semaphore, i)) for i in range(5)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())