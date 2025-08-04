"""Pequeño script para lanzar el servidor gRPC de transcripción."""

from __future__ import annotations

import argparse

from server.models import GRPC_AVAILABLE, load_models, start_grpc_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arranca el servidor gRPC de transcripción")
    parser.add_argument("--port", type=int, default=50051, help="Puerto en el que escuchar")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_models()
    if not GRPC_AVAILABLE:
        raise RuntimeError("gRPC no está disponible. Instale grpcio y grpcio-tools")
    thread = start_grpc_server(port=args.port)
    if thread is not None:
        try:
            thread.join()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
