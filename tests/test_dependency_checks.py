import importlib
import sys
import types
from pathlib import Path

import pytest

# Minimal stubs for heavy dependencies
np_stub = types.SimpleNamespace(
    empty=lambda *a, **k: [],
    float32=float,
    stack=lambda *a, **k: [],
    zeros=lambda *a, **k: [],
    ndarray=object,
)
cv2_stub = types.SimpleNamespace(
    VideoCapture=lambda *a, **k: types.SimpleNamespace(read=lambda: (False, None), release=lambda: None)
)
torch_stub = types.SimpleNamespace(
    cuda=types.SimpleNamespace(is_available=lambda: False),
    hub=types.SimpleNamespace(get_dir=lambda: ""),
    from_numpy=lambda *a, **k: None,
    Tensor=object,
)
pd_stub = types.SimpleNamespace(
    errors=types.SimpleNamespace(EmptyDataError=Exception, ParserError=Exception),
    read_csv=lambda *a, **k: None,
)

sys.modules.setdefault("numpy", np_stub)
sys.modules.setdefault("cv2", cv2_stub)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("pandas", pd_stub)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optical_flow.raft_runner import compute_optical_flow


def _dummy_video(path: Path) -> None:
    path.write_bytes(b"00")


def test_compute_optical_flow_requires_raft(tmp_path, monkeypatch):
    vid = tmp_path / "dummy.mp4"
    _dummy_video(vid)
    monkeypatch.setenv("RAFT_DIR", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        compute_optical_flow(str(vid))


def test_run_openface_warns_when_missing(tmp_path, monkeypatch):
    vid = tmp_path / "dummy.mp4"
    _dummy_video(vid)

    models_stub = types.ModuleType("server.models")
    models_stub.OPENFACE_BIN = None
    models_stub.holistic_model = None
    models_stub.load_models = lambda: None
    models_stub.yolo_model = None
    models_stub.yolox_sess = None
    monkeypatch.setitem(sys.modules, "server.models", models_stub)
    monkeypatch.delitem(sys.modules, "server.feature_extraction", raising=False)

    fe = importlib.import_module("server.feature_extraction")
    with pytest.warns(RuntimeWarning, match="OPENFACE_BIN is not set"):
        assert fe._run_openface(str(vid)) is None
