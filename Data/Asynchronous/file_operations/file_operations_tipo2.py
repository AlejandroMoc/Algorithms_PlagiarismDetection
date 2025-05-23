import asyncio
import aiofiles
import logging

async def async_read_file(filepath):
    async with aiofiles.open(filepath, mode='r') as f:
        file_content = await f.read()
        logging.info(f"Read content: {file_content[:100]}")  # Display first 100 characters
    return file_content

async def main_task():
    await async_read_file('example.txt')
    logging.info("File read successfully")

if __name__ == "__main__":
    asyncio.run(main_task()) 