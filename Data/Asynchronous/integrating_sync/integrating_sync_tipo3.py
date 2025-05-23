import asyncio
import logging

def function_sync():
    logging.info("Running synchronous function")

async def main():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, function_sync)

if __name__ == "__main__":
    asyncio.run(main())