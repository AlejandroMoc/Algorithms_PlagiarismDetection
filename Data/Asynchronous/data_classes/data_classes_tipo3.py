from dataclasses import dataclass
import asyncio
import logging

@dataclass
class WebData:
    url: str
    content: str

async def get_data(url):
    logging.info(f"Fetching data from {url}")
    await asyncio.sleep(1)  # Simulate network delay
    return WebData(url, f"Content from {url}")

async def main():
    web = await get_data("http://example.com")
    logging.info(f"Fetched {web.content} from {web.url}")

if __name__ == "__main__":
    asyncio.run(main())