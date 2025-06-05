import asyncio
import aiohttp
import logging

async 
async 
if __name__ == "__main__":
    asyncio.run(main()) 

def main():
    url = "http://example.com"
    content = await fetch_url(url)
    logging.info(f"Content from {url}: {content[:100]}")  # Print first 100 characters


def fetch_url(url):
    logging.info(f"Fetching URL: {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            result = await response.text()
            logging.info(f"Received data from {url}")
            return result
