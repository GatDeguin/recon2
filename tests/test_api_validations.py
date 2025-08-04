import io
from types import SimpleNamespace

from fastapi.testclient import TestClient

import os, sys, types, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
from server.app import app, MAX_FILE_SIZE, MAX_SEQ_LEN


def _client(monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr("server.app.load_models", lambda: None)
        monkeypatch.setattr("server.app.metrics_logger.log", lambda *a, **k: None)
    else:  # fallback without fixture
        import server.app as app_module

        app_module.load_models = lambda: None
        app_module.metrics_logger.log = lambda *a, **k: None
    return TestClient(app)


def test_rejects_large_file():
    # avoid heavy model loading
    client = _client()
    data = io.BytesIO(b"0" * (MAX_FILE_SIZE + 1))
    resp = client.post("/transcribe", files={"files": ("big.mp4", data, "video/mp4")})
    assert resp.status_code == 413


def test_rejects_wrong_mime():
    client = _client()
    data = io.BytesIO(b"small")
    resp = client.post("/transcribe", files={"files": ("bad.mp4", data, "text/plain")})
    assert resp.status_code == 400
    assert "Unsupported media type" in resp.json()["detail"]


def test_rejects_bad_extension():
    client = _client()
    data = io.BytesIO(b"small")
    resp = client.post("/transcribe", files={"files": ("bad.txt", data, "video/mp4")})
    assert resp.status_code == 400
    assert "Unsupported file extension" in resp.json()["detail"]


def test_rejects_long_sequence(monkeypatch):
    client = _client(monkeypatch)

    def fake_extract(_data: bytes):
        return SimpleNamespace(shape=(1, 1, MAX_SEQ_LEN + 1, 1))

    monkeypatch.setattr("server.app.extract_features_from_bytes", fake_extract)
    data = io.BytesIO(b"0")
    resp = client.post("/transcribe", files={"files": ("seq.mp4", data, "video/mp4")})
    assert resp.status_code == 400
    assert "maximum length" in resp.json()["detail"]


def test_rejects_corrupt_video(monkeypatch):
    client = _client(monkeypatch)

    def raise_extract(_data: bytes):
        raise ValueError("boom")

    monkeypatch.setattr("server.app.extract_features_from_bytes", raise_extract)
    data = io.BytesIO(b"0")
    resp = client.post("/transcribe", files={"files": ("bad.mp4", data, "video/mp4")})
    assert resp.status_code == 400
    assert "corrupt" in resp.json()["detail"].lower()
