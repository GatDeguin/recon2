import sys
from types import SimpleNamespace
import numpy as np
import cv2
import torch
import pytest

# Helper to create a small dummy video

def _make_video(path, frames=2, size=(64, 64)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 5.0, size)
    for _ in range(frames):
        frame = np.full((size[1], size[0], 3), 255, np.uint8)
        out.write(frame)
    out.release()

class DummyTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)
    def cpu(self):
        return self
    def numpy(self):
        return self._arr

class DummyYOLO:
    def __init__(self, *a, **kw):
        pass
    def __call__(self, frame, *a, **kw):
        boxes = SimpleNamespace(xyxy=DummyTensor([[0, 0, frame.shape[1], frame.shape[0]]]))
        return [SimpleNamespace(boxes=boxes)]

class DummyLandmark(SimpleNamespace):
    pass

class DummyHolistic:
    def __init__(self, *a, **kw):
        pass
    def process(self, img):
        def mk(n):
            return SimpleNamespace(landmark=[DummyLandmark(x=0.0, y=0.0, z=0.0) for _ in range(n)])
        return SimpleNamespace(
            pose_landmarks=mk(33),
            left_hand_landmarks=mk(21),
            right_hand_landmarks=mk(21),
            face_landmarks=mk(468),
        )
    def close(self):
        pass

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastapi", SimpleNamespace(FastAPI=lambda: None))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=DummyYOLO))
    mp_mod = SimpleNamespace(holistic=SimpleNamespace(Holistic=DummyHolistic))
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(solutions=mp_mod))
    monkeypatch.setitem(sys.modules, "optical_flow.raft_runner", SimpleNamespace(compute_optical_flow=lambda p: np.empty((0,), np.float32)))


def test_extract_features(tmp_path):
    video = tmp_path / "vid.mp4"
    _make_video(video)
    import server.app as app
    feats = app._extract_features(str(video))
    assert isinstance(feats, torch.Tensor)
    assert feats.shape[2] == 2
    assert feats.shape[3] == 544
