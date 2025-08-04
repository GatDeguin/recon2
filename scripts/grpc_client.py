"""Cliente gRPC sencillo para enviar un vídeo al servidor de transcripción."""

from __future__ import annotations

import argparse
import grpc

from server.protos import transcriber_pb2, transcriber_pb2_grpc


def main() -> None:
    parser = argparse.ArgumentParser(description="Envía un vídeo al servidor gRPC de transcripción")
    parser.add_argument("video", help="Ruta al archivo de vídeo a transcribir")
    parser.add_argument("--host", default="localhost", help="Host del servidor")
    parser.add_argument("--port", type=int, default=50051, help="Puerto del servidor")
    args = parser.parse_args()

    with open(args.video, "rb") as f:
        data = f.read()

    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = transcriber_pb2_grpc.TranscriberStub(channel)
    request = transcriber_pb2.VideoRequest(video=data)
    response = stub.Transcribe(request)
    print(response.transcript)


if __name__ == "__main__":
    main()
