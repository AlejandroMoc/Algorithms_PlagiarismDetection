import asyncio
import logging


async 
if __name__ == "__main__":
    asyncio.run(main()) 

def main():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, synchronous_function)


def synchronous_function():
    logging.info("Running synchronous function")
