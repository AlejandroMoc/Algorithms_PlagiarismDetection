import asyncio
import logging

async 
async 
if __name__ == "__main__":
    asyncio.run(main()) 

def main():
    result = await error_prone_request()
    if result is None:
        logging.info("Handling fallback")
    else:
        logging.info(f"Received: {result}")


def error_prone_request():
    try:
        logging.info("Making a risky request")
        await asyncio.sleep(1)
        raise Exception("Something went wrong!")
    except Exception as e:
        logging.error(f"Caught an error: {e}")
        return None
