"""Simple manual check for the WebSocket endpoint.

Run the backend first:

    uvicorn server.app:app

Then execute this script. It will send a small chunk of the bundled
`demo.mp4` file to ``ws://localhost:8000/ws`` and print the JSON response
from the server.
"""

import asyncio
import json
from pathlib import Path

import websockets


def load_sample_bytes() -> bytes:
    """Return a small portion of demo.mp4 to avoid heavy processing."""
    data = Path(__file__).resolve().parent.parent / "demo.mp4"
    # Send only the first 100 kB to trigger an error but confirm connectivity.
    return data.read_bytes()[:100_000]


async def main():
    uri = "ws://localhost:8000/ws"
    payload = load_sample_bytes()
    async with websockets.connect(uri) as ws:
        await ws.send(payload)
        reply = await ws.recv()
        try:
            data = json.loads(reply)
        except json.JSONDecodeError:
            data = reply
        print("Server replied:", data)


if __name__ == "__main__":
    asyncio.run(main())
