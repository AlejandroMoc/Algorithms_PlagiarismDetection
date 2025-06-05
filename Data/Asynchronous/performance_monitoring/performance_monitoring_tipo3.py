import asyncio
import time
import logging

async 
async 
if __name__ == "__main__":
    asyncio.run(main()) 

def main():
    start_time = time.time()
    await task()
    elapsed_time = time.time() - start_time
    logging.info(f"Task completed in {elapsed_time:.2f} seconds")


def task():
    logging.info("Starting task")
    await asyncio.sleep(1)
    logging.info("Task completed")
