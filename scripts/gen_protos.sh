#!/usr/bin/env bash
set -euo pipefail
python -m grpc_tools.protoc -I server/protos --python_out=server/protos --grpc_python_out=server/protos server/protos/transcriber.proto

