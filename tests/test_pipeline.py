import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest


class DummyTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)
    def cpu(self):
        return self
    def numpy(self):
        return self._arr

class DummyYOLO:
    def __init__(self, *a, **kw):
        self.args = a
        self.kw = kw
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
        self.closed = True


cv2_stub = SimpleNamespace(
    COLOR_BGR2RGB=0,
    cvtColor=lambda img, flag: img,
    resize=lambda img, size: img,
    cuda=SimpleNamespace(
        GpuMat=lambda: SimpleNamespace(upload=lambda *a, **k: None, download=lambda: np.zeros((4, 4, 3), np.uint8)),
        cvtColor=lambda m, f: m,
    ),
)
torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
mp_mod = SimpleNamespace(holistic=SimpleNamespace(Holistic=lambda **kw: DummyHolistic()))
sys.modules.setdefault("cv2", cv2_stub)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("ultralytics", SimpleNamespace(YOLO=DummyYOLO))
sys.modules.setdefault("onnxruntime", SimpleNamespace(InferenceSession=DummySess))
sys.modules.setdefault("mediapipe", SimpleNamespace(solutions=mp_mod))

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import pipeline


@pytest.fixture(autouse=True)
def patch_deps(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=DummyYOLO))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=DummySess))
    mp_mod = SimpleNamespace(holistic=SimpleNamespace(Holistic=lambda **kw: DummyHolistic()))
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(solutions=mp_mod))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))


def test_sys_path_insert():
    expected = os.path.dirname(os.path.dirname(__file__))
    assert sys.path[0] == expected


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


def dummy_model(_feats: np.ndarray) -> np.ndarray:
    return np.array([[[0.0, 1.0]]], np.float32)


def decode(logits: np.ndarray) -> str:
    vocab = {1: "dummy"}
    token = int(np.argmax(logits, axis=-1).squeeze())
    return vocab.get(token, "")


def test_full_pipeline_transcription():
    frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(2)]
    det = pipeline.PersonDetector(conf=0.5)
    ext = pipeline.LandmarkExtractor(mp_conf=0.5)
    feats = []
    for frame in frames:
        box = det.detect(frame)[0]
        lms = ext.extract(frame, box)
        feats.append(
            np.concatenate([
                lms["pose"],
                lms["left_hand"],
                lms["right_hand"],
                lms["face"],
            ])
        )
    ext.close()
    feats = np.stack(feats)
    logits = dummy_model(feats)
    transcript = decode(logits)
    assert transcript == "dummy"
