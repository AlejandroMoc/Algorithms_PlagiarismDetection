import asyncio
import logging

async def audacious_call():
    try:
        logging.info("Making a daring call")
        await asyncio.sleep(1)
        raise Exception("Something went wrong!")
    except Exception as e:
        logging.error(f"Caught an unexpected situation: {e}")
        return None

async def primary():
    outcome = await audacious_call()
    if outcome is None:
        logging.info("Dealing with alternative scenario")
    else:
        logging.info(f"Obtained: {outcome}")

if __name__ == "__main__":
    asyncio.run(primary()) 