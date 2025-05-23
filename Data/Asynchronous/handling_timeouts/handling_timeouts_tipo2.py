import asyncio
import logging

async def retrieve_with_limit(site, limit_time):
    logging.info(f"Retrieving {site} within a limit of {limit_time} seconds")
    try:
        await asyncio.wait_for(asyncio.sleep(3), timeout=limit_time)  # Simulate a delay that might exceed timeout
        logging.info("Retrieval successful")
        return f"Info from {site}"
    except asyncio.TimeoutError:
        logging.error("Retrieval timed out")
        return None

async def primary_function():
    result = await retrieve_with_limit("http://example.com", 2)
    if result is None:
        logging.info("Operation timed out")
    else:
        logging.info(f"Received: {result}")

if __name__ == "__main__":
    asyncio.run(primary_function())