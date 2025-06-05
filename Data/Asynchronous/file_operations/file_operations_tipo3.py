import asyncio
import aiofiles
import logging

async 
async 
if __name__ == "__main__":
    asyncio.run(main()) 

def main():
    await read_file_async('example.txt')
    logging.info("File read successfully")


def read_file_async(camino):
    async with aiofiles.open(camino, mode='r') as file:
        content = await file.read()
        logging.info(f"Read content: {content[:100]}")  # Display first 100 characters
    return content
