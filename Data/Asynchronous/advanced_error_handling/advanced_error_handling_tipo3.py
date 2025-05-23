import asyncio
import logging

async def request_with_errors():
    try:
        logging.info("Making a risky request")
        await asyncio.sleep(1)
        raise Exception("Something went wrong!")
    except Exception as exc:
        logging.error(f"Caught an error: {exc}")
        return None

async def main_program():
    result = await request_with_errors()
    if result is None:
        logging.info("Handling fallback")
    else:
        logging.info(f"Received: {result}")

if __name__ == "__main__":
    asyncio.run(main_program()) 