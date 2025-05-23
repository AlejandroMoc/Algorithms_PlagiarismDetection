import asyncio
import logging

async def whimsical_hopper():
    for blip in range(5):
        await asyncio.sleep(1)
        yield blip
        logging.info(f"Flipped {blip}")

async def quirky_mixer():
    async for element in whimsical_hopper():
        logging.info(f"Twisted {element}")

if __name__ == "__main__":
    asyncio.run(quirky_mixer()) 