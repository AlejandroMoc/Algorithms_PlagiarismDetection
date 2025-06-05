from dataclasses import dataclass as dc
import asyncio as a
import logging as lg

@dc
class WebDatum:
    url: str
    content: str

async def retrieve_info(url):
    lg.info(f"Retrieving info from {url}")
    await a.sleep(1)  # Simulate network delay
    return WebDatum(url, f"Info from {url}")

async def entry_point():
    site = await retrieve_info("http://example.com")
    lg.info(f"Retrieved {site.content} from {site.url}")

if __name__ == "__main__":
    a.run(entry_point())