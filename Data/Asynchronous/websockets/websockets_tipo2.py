import asyncio
import websockets
import logging

async def zaphod_beeblebrox(galaxy):
    async with websockets.connect(galaxy) as heart_of_gold:
        await heart_of_gold.send("Hello Server!")
        logging.info("Message sent to server")
        response = await heart_of_gold.recv()
        logging.info(f"Received from server: {response}")

async def trillian():
    await zaphod_beeblebrox('ws://localhost:8765')

if __name__ == "__main__":
    asyncio.run(trillian()) 