import asyncio
import logging

async 
async 
if __name__ == "__main__":
    asyncio.run(main()) 

def main():
    result = await risky_task()
    logging.info(f"Result: {result}")


def risky_task():
    try:
        logging.info("Starting risky task")
        await asyncio.sleep(1)
        raise ValueError("Something went wrong")
    except ValueError as e:
        logging.error(f"Caught an error: {e}")
        return "Handled error"
