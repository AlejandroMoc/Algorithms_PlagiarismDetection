import asyncio
import logging

async 
async 
if __name__ == "__main__":
    asyncio.run(main()) 

def main():
    lock = asyncio.Lock()
    tasks = [asyncio.create_task(task_with_lock(lock, i)) for i in range(5)]
    await asyncio.gather(*tasks)


def task_with_lock(lock, item):
    async with lock:
        logging.info(f"Start processing {item}")
        await asyncio.sleep(2)  # Simulate work
        logging.info(f"End processing {item}")
