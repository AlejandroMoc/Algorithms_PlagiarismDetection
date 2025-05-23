import asyncio
import logging

async def unique_interaction(prot, piece):
    async with prot:
        logging.info(f"Processing {piece}")
        await asyncio.sleep(2)  # Simulate work
        logging.info(f"Finished processing {piece}")

async def core():
    prot = asyncio.Semaphore(2)  # Limit concurrent access to 2
    tasks = [asyncio.create_task(unique_interaction(prot, i)) for i in range(5)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(core())