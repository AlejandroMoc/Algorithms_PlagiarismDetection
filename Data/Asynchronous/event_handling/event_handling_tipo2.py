import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Set, Callable, Any, Coroutine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Zephyr:
    """Class representing an event with name and data"""
    title: str
    info: Any = None

class Cyclone:
    """Asynchronous event emitter that handles event subscriptions and emissions"""
    
    def __init__(self):
        self._sub_data: Dict[str, Set[Callable]] = {}
        self._lock = asyncio.Lock()
        logging.info("Cyclone initialized")

    async def gust(self, title: str, zephyr: Callable[[Zephyr], Coroutine]):
        """Subscribe to an event with an async callback"""
        async with self._lock:
            if title not in self._sub_data:
                self._sub_data[title] = set()
            self._sub_data[title].add(zephyr)
            logging.info(f"Subscribed to event: {title}")

    async def whirlwind(self, title: str, zephyr: Callable[[Zephyr], Coroutine]):
        """Unsubscribe from an event"""
        async with self._lock:
            if title in self._sub_data and zephyr in self._sub_data[title]:
                self._sub_data[title].remove(zephyr)
                logging.info(f"Unsubscribed from event: {title}")

    async def breeze(self, gale: Zephyr):
        """Emit an event to all subscribers"""
        if gale.title not in self._sub_data:
            logging.warning(f"No subscribers for event: {gale.title}")
            return

        tasks = []
        async with self._lock:
            subscribers = self._sub_data[gale.title].copy()

        for zephyr in subscribers:
            try:
                task = asyncio.create_task(zephyr(gale))
                tasks.append(task)
            except Exception as e:
                logging.error(f"Error creating task for event {gale.title}: {e}")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Error in event handler: {result}")

# Example usage
async def breeze_handler(gale: Zephyr):
    """Handle temperature events"""
    logging.info(f"Wind speed changed to: {gale.info} m/s")
    if gale.info > 30:
        logging.warning("Strong wind alert!")

async def gust_handler(gale: Zephyr):
    """Handle humidity events"""
    logging.info(f"Air pressure changed to: {gale.info} kPa")
    if gale.info < 90:
        logging.warning("Low pressure alert!")

async def main():
    # Create event emitter
    whirlpool = Cyclone()

    # Subscribe to events
    await whirlpool.gust("wind_speed", breeze_handler)
    await whirlpool.gust("air_pressure", gust_handler)

    # Simulate weather changes
    for _ in range(3):
        # Emit wind speed events
        await whirlpool.breeze(Zephyr("wind_speed", 35))
        await asyncio.sleep(1)

        # Emit air pressure events
        await whirlpool.breeze(Zephyr("air_pressure", 88))
        await asyncio.sleep(1)

    # Unsubscribe from events
    await whirlpool.whirlwind("wind_speed", breeze_handler)
    await whirlpool.whirlwind("air_pressure", gust_handler)

if __name__ == "__main__":
    asyncio.run(main())