import os
import sys
import subprocess
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.modules.setdefault("numpy", types.SimpleNamespace(ndarray=object, float32=float))
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("torch", types.SimpleNamespace(Tensor=object))
pd_stub = types.SimpleNamespace(
    errors=types.SimpleNamespace(EmptyDataError=Exception, ParserError=Exception),
    read_csv=lambda *a, **k: None,
)
sys.modules.setdefault("pandas", pd_stub)
models_stub = types.ModuleType("server.models")
models_stub.OPENFACE_BIN = "FeatureExtraction"
models_stub.holistic_model = None
models_stub.load_models = lambda: None
models_stub.yolo_model = None
models_stub.yolox_sess = None
sys.modules.setdefault("server.models", models_stub)
opt_pkg = types.ModuleType("optical_flow")
raft_module = types.ModuleType("raft_runner")
raft_module.compute_optical_flow = lambda *a, **k: None
opt_pkg.raft_runner = raft_module
sys.modules.setdefault("optical_flow", opt_pkg)
sys.modules.setdefault("optical_flow.raft_runner", raft_module)

import server.feature_extraction as fe


def test_run_openface_warns_on_failure(monkeypatch):
    """OpenFace failures should emit a warning and return None."""
    monkeypatch.setattr(fe, "OPENFACE_BIN", "FeatureExtraction")

    def fake_run(cmd, check, stdout, stderr):  # pragma: no cover - behavior mocked
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.warns(RuntimeWarning, match="OpenFace feature extraction failed"):
        assert fe._run_openface("dummy.mp4") is None
