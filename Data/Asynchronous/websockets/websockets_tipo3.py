import asyncio
import websockets
import logging

async def send_receive_websocket_message(uri):
    async with websockets.connect(uri) as ws:
        await ws.send("Hello Server!")
        logging.info("Message sent to server")
        response = await ws.recv()
        logging.info(f"Received from server: {response}")

async def main_program():
    await send_receive_websocket_message('ws://localhost:8765')

if __name__ == "__main__":
    asyncio.run(main_program())