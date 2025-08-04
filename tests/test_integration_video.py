import os
import sys
from types import SimpleNamespace
import numpy as np

# Dummy modules to avoid heavy dependencies
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
    def __init__(self, *a, **kw):
        self.args = a
    def get_inputs(self):
        return [SimpleNamespace(shape=[1, 3, 4, 4], name="in")]
    def run(self, *_a, **_kw):
        return [np.zeros((1, 6), np.float32)]

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

class DummyGpuMat:
    def upload(self, *a, **k):
        self.uploaded = True
    def download(self):
        return np.zeros((4, 4, 3), np.uint8)

cv2_stub = SimpleNamespace(
    COLOR_BGR2RGB=0,
    cvtColor=lambda img, flag: img,
    resize=lambda img, size: img,
    VideoCapture=lambda path: SimpleNamespace(read=lambda: (False, None)),
    cuda=SimpleNamespace(GpuMat=lambda: DummyGpuMat(), cvtColor=lambda m, f: m),
)

sys.modules.setdefault("cv2", cv2_stub)
sys.modules.setdefault("ultralytics", SimpleNamespace(YOLO=DummyYOLO))
sys.modules.setdefault("onnxruntime", SimpleNamespace(InferenceSession=DummySess))
mp_mod = SimpleNamespace(holistic=SimpleNamespace(Holistic=lambda **kw: DummyHolistic()))
sys.modules.setdefault("mediapipe", SimpleNamespace(solutions=mp_mod))
sys.modules.setdefault("torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import pipeline


def dummy_model(_feats: np.ndarray) -> np.ndarray:
    return np.array([[[0.0, 1.0]]], np.float32)


def decode(logits: np.ndarray) -> str:
    vocab = {1: "dummy"}
    token = int(np.argmax(logits, axis=-1).squeeze())
    return vocab.get(token, "")


def test_integration_pipeline():
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
    assert feats.shape == (2, 1629)
    logits = dummy_model(feats)
    transcript = decode(logits)
    assert transcript == "dummy"
