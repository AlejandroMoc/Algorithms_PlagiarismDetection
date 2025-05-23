import asyncio
import logging

class CustomAsyncManager:
    async def __aenter__(self):
        logging.info("Enter context")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        logging.info("Exit context")

async def custom_function():
    async with CustomAsyncManager():
        logging.info("Doing work inside the context")

if __name__ == "__main__":
    asyncio.run(custom_function()) 