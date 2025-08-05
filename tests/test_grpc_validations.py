import os, sys, types, time, numpy as np, grpc, pytest
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "server", "protos"))

torch_stub = types.SimpleNamespace(nn=types.SimpleNamespace(Module=object), Tensor=object)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("uvicorn", types.SimpleNamespace())
sys.modules.setdefault(
    "mediapipe",
    types.SimpleNamespace(solutions=types.SimpleNamespace(holistic=types.SimpleNamespace(Holistic=lambda **k: None))),
)
sys.modules.setdefault("ultralytics", types.SimpleNamespace(YOLO=lambda *a, **k: None))
sys.modules.setdefault("onnxruntime", types.SimpleNamespace(InferenceSession=lambda *a, **k: None))
sys.modules.setdefault("infer", types.SimpleNamespace(beam_search=lambda *a, **k: []))
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("pandas", types.SimpleNamespace())
opt_pkg = types.ModuleType("optical_flow")
raft_module = types.ModuleType("raft_runner")
raft_module.compute_optical_flow = lambda *a, **k: np.zeros((0,), dtype=float)
opt_pkg.raft_runner = raft_module
sys.modules.setdefault("optical_flow", opt_pkg)
sys.modules.setdefault("optical_flow.raft_runner", raft_module)
models_pkg = types.ModuleType("models")
transformer_lm_module = types.ModuleType("transformer_lm")
transformer_lm_module.load_model = lambda *a, **k: None
models_pkg.transformer_lm = transformer_lm_module
sys.modules.setdefault("models", models_pkg)
sys.modules.setdefault("models.transformer_lm", transformer_lm_module)

from server.models import start_grpc_server, MAX_FILE_SIZE, MAX_SEQ_LEN
from server.protos import transcriber_pb2, transcriber_pb2_grpc


def _make_stub(monkeypatch, seq_len, port):
    def fake_extract(_data: bytes):
        return SimpleNamespace(shape=(1, 1, seq_len))

    monkeypatch.setattr(
        "server.feature_extraction.extract_features_from_bytes", fake_extract
    )
    monkeypatch.setattr("server.models.decode", lambda logits: "")
    monkeypatch.setattr("server.models.onnx_sess", None)
    monkeypatch.setattr("server.models.model", None)
    start_grpc_server(port=port)
    time.sleep(0.1)
    channel = grpc.insecure_channel(f"localhost:{port}")
    return transcriber_pb2_grpc.TranscriberStub(channel)


def test_grpc_rejects_large_file(monkeypatch):
    stub = _make_stub(monkeypatch, seq_len=1, port=50060)
    data = b"0" * (MAX_FILE_SIZE + 1)
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.Transcribe(transcriber_pb2.VideoRequest(video=data))
    assert excinfo.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED


def test_grpc_rejects_long_sequence(monkeypatch):
    stub = _make_stub(monkeypatch, seq_len=MAX_SEQ_LEN + 1, port=50061)
    data = b"0"
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.Transcribe(transcriber_pb2.VideoRequest(video=data))
    assert excinfo.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
