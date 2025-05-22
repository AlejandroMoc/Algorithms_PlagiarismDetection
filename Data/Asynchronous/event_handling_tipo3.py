import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Set, Callable, Any, Coroutine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Event:
    name: str
    data: Any = None

class EventNotifier:
    
    def __init__(self):
        self._listeners: Dict[str, Set[Callable]] = {}
        self._lock = asyncio.Lock()
        logging.info("EventNotifier initialized")

    async def register(self, event_name: str, callback: Callable[[Event], Coroutine]):
        async with self._lock:
            if event_name not in self._listeners:
                self._listeners[event_name] = set()
            self._listeners[event_name].add(callback)
            logging.info(f"Registered for event: {event_name}")

    async def unregister(self, event_name: str, callback: Callable[[Event], Coroutine]):
        async with self._lock:
            if event_name in self._listeners and callback in self._listeners[event_name]:
                self._listeners[event_name].remove(callback)
                logging.info(f"Unregistered from event: {event_name}")

    async def notify(self, event: Event):
        if event.name not in self._listeners:
            logging.warning(f"No listeners for event: {event.name}")
            return

        tasks = []
        async with self._lock:
            listeners = self._listeners[event.name].copy()

        for callback in listeners:
            try:
                task = asyncio.create_task(callback(event))
                tasks.append(task)
            except Exception as e:
                logging.error(f"Error creating task for event {event.name}: {e}")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Error in event handler: {result}")

async def temp_handler(event: Event):
    logging.info(f"Temperature changed to: {event.data}°C")
    if event.data > 30:
        logging.warning("High temperature alert!")

async def humidity_handler(event: Event):
    logging.info(f"Humidity changed to: {event.data}%")
    if event.data > 80:
        logging.warning("High humidity alert!")

async def run():
    notifier = EventNotifier()

    await notifier.register("temperature", temp_handler)
    await notifier.register("humidity", humidity_handler)

    for _ in range(3):
        await notifier.notify(Event("temperature", 32))
        await asyncio.sleep(1)

        await notifier.notify(Event("humidity", 85))
        await asyncio.sleep(1)

    await notifier.unregister("temperature", temp_handler)
    await notifier.unregister("humidity", humidity_handler)

if __name__ == "__main__":
    asyncio.run(run())