# gRPC Protocol Buffers

Este directorio contiene la definición del servicio gRPC.

Para regenerar los *stubs* de Python (`transcriber_pb2.py` y `transcriber_pb2_grpc.py`) ejecute:

```bash
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. transcriber.proto
```

El comando debe ejecutarse desde este mismo directorio o ajustando las rutas de búsqueda con `-I`.
