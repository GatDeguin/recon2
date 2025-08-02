"""Reusable components for video preprocessing pipeline."""

from __future__ import annotations

from typing import Tuple, List, Dict

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

try:  # pragma: no cover - optional dependency
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

try:  # pragma: no cover - optional dependency
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None

try:  # pragma: no cover - optional dependency
    import ruptures as rpt
except Exception:  # pragma: no cover
    rpt = None

import torch  # pragma: no cover - torch is required by some detectors


class PersonDetector:
    """Detect people using YOLOv8 or YOLOX."""

    def __init__(self, conf: float = 0.5, yolox_model: str | None = None) -> None:
        self.conf = conf
        if yolox_model:
            if ort is None:
                raise RuntimeError("onnxruntime is not available")
            providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(yolox_model, providers=providers)
            self.model = None
        else:
            if YOLO is None:
                raise RuntimeError("ultralytics is not available")
            self.model = YOLO("yolov8n.pt")
            self.session = None

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return ``Nx4`` array of bounding boxes ``(x1, y1, x2, y2)``."""
        if self.session is not None:
            ih, iw = self.session.get_inputs()[0].shape[2:]
            img = cv2.resize(frame, (iw, ih)).astype(np.float32)
            img = img.transpose(2, 0, 1)[None]
            preds = self.session.run(None, {self.session.get_inputs()[0].name: img})[0]
            if preds.ndim == 3:
                preds = preds[0]
            valid = preds[(preds[:, 4] >= self.conf) & (preds[:, 5] == 0)]
            scale_x = frame.shape[1] / iw
            scale_y = frame.shape[0] / ih
            boxes = valid[:, :4].copy()
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y
            return boxes.astype(int)
        else:
            yres = self.model(frame, device=0, half=True, conf=self.conf, classes=[0])[0]
            return yres.boxes.xyxy.cpu().numpy().astype(int)


class LandmarkExtractor:
    """Extract pose, hand and face landmarks using MediaPipe Holistic."""

    def __init__(self, mp_conf: float = 0.7) -> None:
        if mp is None:
            raise RuntimeError("mediapipe is not available")
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=mp_conf,
            min_tracking_confidence=mp_conf,
        )

    def extract(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> Dict[str, np.ndarray]:
        """Return landmark arrays in global coordinates."""
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        try:
            gpu_roi = cv2.cuda_GpuMat(); gpu_roi.upload(roi)
            gpu_rgb = cv2.cuda.cvtColor(gpu_roi, cv2.COLOR_BGR2RGB)
            rgb = gpu_rgb.download()
        except Exception:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = self.holistic.process(rgb)

        def lm_arr(lm, n):
            return np.array([[p.x, p.y, p.z] for p in lm.landmark], np.float32) if lm else np.zeros((n, 3), np.float32)

        pose_lm = lm_arr(getattr(res, "pose_landmarks", None), 33)
        lh_lm = lm_arr(getattr(res, "left_hand_landmarks", None), 21)
        rh_lm = lm_arr(getattr(res, "right_hand_landmarks", None), 21)
        face_lm = lm_arr(getattr(res, "face_landmarks", None), 468)

        def to_global(pts: np.ndarray) -> np.ndarray:
            g = pts.copy()
            g[:, 0] = g[:, 0] * (x2 - x1) + x1
            g[:, 1] = g[:, 1] * (y2 - y1) + y1
            return g

        return {
            "pose": to_global(pose_lm).reshape(-1),
            "left_hand": to_global(lh_lm).reshape(-1),
            "right_hand": to_global(rh_lm).reshape(-1),
            "face": to_global(face_lm).reshape(-1),
            "lh_center": tuple(to_global(lh_lm)[:, :2].mean(axis=0)),
            "rh_center": tuple(to_global(rh_lm)[:, :2].mean(axis=0)),
        }

    def close(self) -> None:
        self.holistic.close()


def segment_sequences(lh_seq: List[np.ndarray], rh_seq: List[np.ndarray], face_seq: List[np.ndarray], penalty: int = 10) -> List[Tuple[int, int]]:
    """Detect change points on landmark velocities and return segments."""
    if len(lh_seq) == 0:
        return [(0, 0)]

    lh = np.stack(lh_seq)
    rh = np.stack(rh_seq)
    face = np.stack(face_seq)
    speed = np.zeros(len(lh), np.float32)
    if len(lh) > 1:
        lh_diff = np.linalg.norm(lh[1:] - lh[:-1], axis=1)
        rh_diff = np.linalg.norm(rh[1:] - rh[:-1], axis=1)
        face_diff = np.linalg.norm(face[1:] - face[:-1], axis=1)
        speed[1:] = lh_diff + rh_diff + 0.1 * face_diff

    if rpt is not None and len(speed) > 1:
        algo = rpt.Pelt(model="rbf").fit(speed.reshape(-1, 1))
        boundaries = [0] + algo.predict(pen=penalty)
    else:
        thr = speed.mean() * 0.5
        boundaries = [0]
        for i in range(1, len(speed)):
            if speed[i] < thr and speed[max(0, i - 1)] >= thr:
                boundaries.append(i)
        boundaries.append(len(speed))

    segments: List[Tuple[int, int]] = []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if e > s:
            segments.append((s, e))
    if not segments:
        segments.append((0, len(speed)))
    return segments
