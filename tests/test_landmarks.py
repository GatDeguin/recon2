import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest


class DummyLandmark(SimpleNamespace):
    pass


class DummyHolistic:
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
        self.closed = True


cv2_stub = SimpleNamespace(
    COLOR_BGR2RGB=0,
    cvtColor=lambda img, flag: img,
    cuda=SimpleNamespace(
        GpuMat=lambda: SimpleNamespace(upload=lambda *a, **k: None, download=lambda: np.zeros((4, 4, 3), np.uint8)),
        cvtColor=lambda m, f: m,
    ),
)
torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
mp_mod = SimpleNamespace(holistic=SimpleNamespace(Holistic=lambda **kw: DummyHolistic()))
sys.modules.setdefault("cv2", cv2_stub)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("mediapipe", SimpleNamespace(solutions=mp_mod))

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import pipeline


def test_landmark_shapes_and_types():
    ext = pipeline.LandmarkExtractor(mp_conf=0.5)
    frame = np.zeros((4, 4, 3), np.uint8)
    data = ext.extract(frame, (0, 0, 4, 4))

    expected = {"pose": 33, "left_hand": 21, "right_hand": 21, "face": 468}
    for key, n in expected.items():
        arr = data[key]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (n * 3,)
        assert arr.dtype == np.float32

    ext.close()

