import asyncio
import json
import time

import websockets


async def _echo_server(websocket):
    async for message in websocket:
        # Simulate small processing delay
        await asyncio.sleep(0.01)
        await websocket.send(json.dumps({"transcript": "ok"}))


def test_websocket_latency_fps():
    async def run_test():
        server = await websockets.serve(_echo_server, "localhost", 8765)
        try:
            async with websockets.connect("ws://localhost:8765") as ws:
                payload = b"0" * 5
                start = time.time()
                await ws.send(payload)
                resp = await ws.recv()
                end = time.time()
                latency = end - start
                fps = len(payload) / latency if latency > 0 else 0.0
                data = json.loads(resp)
                assert data["transcript"] == "ok"
                assert latency > 0
                assert fps > 0
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run_test())
