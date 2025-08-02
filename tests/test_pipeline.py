import sys
from types import SimpleNamespace

import numpy as np
import pytest

from utils import pipeline


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

class DummySess:
    def __init__(self, path, providers=None):
        self._input = SimpleNamespace(shape=[1, 3, 4, 4], name="in")
    def get_inputs(self):
        return [self._input]
    def run(self, *_a, **_kw):
        return [np.array([[[0, 0, 2, 2, 0.6, 0]]], np.float32)]

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
        pass


@pytest.fixture(autouse=True)
def patch_deps(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=DummyYOLO))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=DummySess))
    mp_mod = SimpleNamespace(holistic=SimpleNamespace(Holistic=lambda **kw: DummyHolistic()))
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(solutions=mp_mod))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))


def test_person_detector_yolo():
    det = pipeline.PersonDetector(conf=0.5)
    frame = np.zeros((4, 4, 3), np.uint8)
    boxes = det.detect(frame)
    assert boxes.shape == (1, 4)
    assert (boxes[0] == [0, 0, 4, 4]).all()


def test_person_detector_yolox():
    det = pipeline.PersonDetector(conf=0.5, yolox_model="model.onnx")
    frame = np.zeros((4, 4, 3), np.uint8)
    boxes = det.detect(frame)
    assert boxes.shape[0] == 1


def test_landmark_extractor():
    ext = pipeline.LandmarkExtractor(mp_conf=0.5)
    frame = np.zeros((4, 4, 3), np.uint8)
    data = ext.extract(frame, (0, 0, 4, 4))
    assert data["pose"].shape == (33 * 3,)
    assert data["left_hand"].shape == (21 * 3,)
    assert data["right_hand"].shape == (21 * 3,)
    assert data["face"].shape == (468 * 3,)
    ext.close()


def test_segment_sequences_no_change(monkeypatch):
    monkeypatch.setattr(pipeline, "rpt", None)
    zeros = [np.zeros(3) for _ in range(3)]
    segs = pipeline.segment_sequences(zeros, zeros, zeros)
    assert segs == [(0, 3)]
