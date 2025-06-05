from dataclasses import dataclass
import asyncio
import logging

@dataclass
class WebsiteData:
    url: str
    content: str

async 
async 
if __name__ == "__main__":
    asyncio.run(main())


def main():
    website = await fetch_data("http://example.com")
    logging.info(f"Fetched {website.content} from {website.url}")


def fetch_data(url):
    logging.info(f"Fetching data from {url}")
    await asyncio.sleep(1)  # Simulate network delay
    return WebsiteData(url, f"Content from {url}")
