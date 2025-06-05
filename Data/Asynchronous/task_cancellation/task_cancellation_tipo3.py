import asyncio
import logging

async 
async 
if __name__ == "__main__":
    asyncio.run(main()) 

def main():
    task = asyncio.create_task(cancellable_task())
    await asyncio.sleep(1)  # Let the task inicio
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logging.info("Main noticed that the task was cancelled")


def cancellable_task():
    try:
        logging.info("Task started")
        await asyncio.sleep(10)  # Long running task
        logging.info("Task completed")
    except asyncio.CancelledError:
        logging.info("Task was cancelled")
        raise
