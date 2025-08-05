import os
import sys
import types
import grpc
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "server", "protos"))


def _dummy_feats(length):
    return types.SimpleNamespace(shape=(1, 3, length, 1))


# Stub heavy optional dependencies before importing server modules
sys.modules.setdefault(
    "mediapipe",
    types.SimpleNamespace(solutions=types.SimpleNamespace(holistic=types.SimpleNamespace(Holistic=lambda **kw: None))),
)
sys.modules.setdefault("ultralytics", types.SimpleNamespace(YOLO=lambda *a, **k: None))
sys.modules.setdefault("onnxruntime", types.SimpleNamespace(InferenceSession=lambda *a, **k: None))
# Minimal decoder stub to avoid heavy dependencies
dec_stub = types.ModuleType("decoder")
dec_stub.init_decoder = lambda *a, **k: None
dec_stub.decode = lambda *_a, **_k: ""
sys.modules["server.decoder"] = dec_stub

# Provide a lightweight feature_extraction module
fe_stub = types.ModuleType("feature_extraction")
fe_stub.extract_features_from_bytes = lambda data: _dummy_feats(1)
sys.modules["server.feature_extraction"] = fe_stub

from server.models import start_grpc_server
from server.protos import transcriber_pb2, transcriber_pb2_grpc


def _make_stub(port):
    channel = grpc.insecure_channel(f"localhost:{port}")
    grpc.channel_ready_future(channel).result(timeout=5)
    stub = transcriber_pb2_grpc.TranscriberStub(channel)
    return stub, channel


def test_transcribe_success(monkeypatch):
    import server.feature_extraction as fe

    monkeypatch.setattr(fe, "extract_features_from_bytes", lambda data: _dummy_feats(1))
    port = 50055
    start_grpc_server(port)
    monkeypatch.setenv("MAX_VIDEO_BYTES", "1000")
    monkeypatch.setenv("MAX_SEQUENCE_LENGTH", "10")
    stub, channel = _make_stub(port)
    resp = stub.Transcribe(transcriber_pb2.VideoRequest(video=b"abc"))
    assert isinstance(resp.transcript, str)
    channel.close()


def test_video_size_limit(monkeypatch):
    import server.feature_extraction as fe

    monkeypatch.setattr(fe, "extract_features_from_bytes", lambda data: _dummy_feats(1))
    port = 50056
    start_grpc_server(port)
    monkeypatch.setenv("MAX_VIDEO_BYTES", "1")
    stub, channel = _make_stub(port)
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.Transcribe(transcriber_pb2.VideoRequest(video=b"toolarge"))
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    channel.close()


def test_sequence_length_limit(monkeypatch):
    import server.feature_extraction as fe

    monkeypatch.setattr(fe, "extract_features_from_bytes", lambda data: _dummy_feats(5))
    port = 50057
    start_grpc_server(port)
    monkeypatch.setenv("MAX_SEQUENCE_LENGTH", "1")
    stub, channel = _make_stub(port)
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.Transcribe(transcriber_pb2.VideoRequest(video=b"a"))
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    channel.close()
