import asyncio
import logging

async def retrieve_info(site):
    try:
        logging.info(f"Attempting to acquire information from {site}")
        await asyncio.sleep(1)  # Simulate a network delay
        if site == "http://error.com":
            raise ValueError("Invalid URL")
        return f"Info from {site}"
    except ValueError as e:
        logging.error(f"Error acquiring information: {e}")
        return None

async def core_operation():
    sites = ["http://example.com", "http://error.com"]
    for site in sites:
        result = await retrieve_info(site)
        if result:
            logging.info(f"Successfully obtained information: {result}")
        else:
            logging.info("Failed to acquire information")

if __name__ == "__main__":
    asyncio.run(core_operation())