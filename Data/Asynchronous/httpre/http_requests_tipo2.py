import asyncio
import aiohttp
import logging

async def retrieve_data(address):
    logging.info(f"Accessing Address: {address}")
    async with aiohttp.ClientSession() as session:
        async with session.get(address) as response:
            result = await response.text()
            logging.info(f"Data retrieved from {address}")
            return result

async def core_process():
    address = "http://example.com"
    data = await retrieve_data(address)
    logging.info(f"Data from {address}: {data[:100]}")  # Print first 100 characters

if __name__ == "__main__":
    asyncio.run(core_process())