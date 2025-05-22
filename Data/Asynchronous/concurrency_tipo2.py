import asyncio
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def custom_task(identifier, wait_time):
    """Simulates an async task with a delay"""
    logging.info(f"Job {identifier} starting")
    await asyncio.sleep(wait_time)
    logging.info(f"Job {identifier} finished")
    return f"Output from job {identifier}"

async def primary_function():
    # Create multiple tasks to run concurrently
    jobs = [
        asyncio.create_task(custom_task("X", 2)),
        asyncio.create_task(custom_task("Y", 1)),
        asyncio.create_task(custom_task("Z", 3))
    ]
    
    # Wait for all tasks to complete
    outputs = await asyncio.gather(*jobs)
    
    # Process results
    for output in outputs:
        logging.info(output)

if __name__ == "__main__":
    asyncio.run(primary_function())