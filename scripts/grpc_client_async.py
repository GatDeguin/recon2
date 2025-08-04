"""Ejemplo de cliente gRPC asíncrono."""

from __future__ import annotations

import argparse
import asyncio
import grpc

from server.protos import transcriber_pb2, transcriber_pb2_grpc


async def main() -> None:
    parser = argparse.ArgumentParser(description="Cliente gRPC asíncrono para transcripción")
    parser.add_argument("video", help="Ruta al archivo de vídeo")
    parser.add_argument("--host", default="localhost", help="Host del servidor")
    parser.add_argument("--port", type=int, default=50051, help="Puerto del servidor")
    args = parser.parse_args()

    with open(args.video, "rb") as f:
        data = f.read()

    channel = grpc.aio.insecure_channel(f"{args.host}:{args.port}")
    stub = transcriber_pb2_grpc.TranscriberStub(channel)
    response = await stub.Transcribe(transcriber_pb2.VideoRequest(video=data))
    print(response.transcript)
    await channel.close()


if __name__ == "__main__":
    asyncio.run(main())
