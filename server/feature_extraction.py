import os
import tempfile
import subprocess
import shutil
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch

from optical_flow.raft_runner import compute_optical_flow
from .models import OPENFACE_BIN, holistic_model, yolo_model, yolox_sess


def _run_openface(path: str) -> Optional[np.ndarray]:
    """Run OpenFace FeatureExtraction on *path* returning [Rx,Ry,Rz,AUs]."""
    if not OPENFACE_BIN:
        return None
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, "of.csv")
    cmd = [OPENFACE_BIN, "-f", path, "-aus", "-pose", "-of", csv_path, "-q"]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        df = pd.read_csv(csv_path)
        head = df[["pose_Rx", "pose_Ry", "pose_Rz"]].to_numpy(np.float32)
        au_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]
        aus = df[au_cols].to_numpy(np.float32)
        return np.concatenate([head, aus], axis=1)
    except Exception:
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _torso_tilt(pose_lm: np.ndarray) -> np.ndarray:
    """Approximate torso orientation from pose landmarks."""
    try:
        ls, rs = pose_lm[11], pose_lm[12]
        lh, rh = pose_lm[23], pose_lm[24]
    except Exception:
        return np.zeros(3, np.float32)
    shoulders = (ls + rs) / 2
    hips = (lh + rh) / 2
    vec = shoulders - hips
    pitch = np.arctan2(vec[2], np.linalg.norm(vec[:2]))
    yaw = np.arctan2(rs[1] - ls[1], rs[0] - ls[0])
    roll = 0.0
    return np.array([pitch, yaw, roll], np.float32)


def extract_features_from_bytes(data: bytes) -> torch.Tensor:
    """Generate ST-GCN features from raw video bytes."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        return _extract_features(tmp.name)


def _extract_features(path: str) -> torch.Tensor:
    """Compute landmarks and optical flow for a video path."""
    cap = cv2.VideoCapture(path)
    of_feats = _run_openface(path)
    n_aus = of_feats.shape[1] - 3 if of_feats is not None else 0
    pose_seq, lh_seq, rh_seq, face_seq = [], [], [], []
    head_seq, torso_seq, au_seq = [], [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if yolox_sess is not None:
            ih, iw = yolox_sess.get_inputs()[0].shape[2:]
            img = cv2.resize(frame, (iw, ih)).astype(np.float32)
            img = img.transpose(2, 0, 1)[None]
            preds = yolox_sess.run(None, {yolox_sess.get_inputs()[0].name: img})[0]
            if preds.ndim == 3:
                preds = preds[0]
            valid = preds[(preds[:, 4] >= 0.5) & (preds[:, 5] == 0)]
            scale_x = frame.shape[1] / iw
            scale_y = frame.shape[0] / ih
            boxes = valid[:, :4].copy()
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y
            boxes = boxes.astype(int)
        else:
            yres = yolo_model(
                frame,
                device=0 if torch.cuda.is_available() else "cpu",
                half=torch.cuda.is_available(),
                conf=0.5,
                classes=[0],
            )[0]
            boxes = yres.boxes.xyxy.cpu().numpy().astype(int)
        if len(boxes) == 0:
            pose_seq.append(np.zeros((33 * 3,), np.float32))
            lh_seq.append(np.zeros((21 * 3,), np.float32))
            rh_seq.append(np.zeros((21 * 3,), np.float32))
            face_seq.append(np.zeros((468 * 3,), np.float32))
            continue
        x1, y1, x2, y2 = boxes[0]
        roi = frame[y1:y2, x1:x2]
        try:
            gpu_roi = cv2.cuda_GpuMat()
            gpu_roi.upload(roi)
            gpu_rgb = cv2.cuda.cvtColor(gpu_roi, cv2.COLOR_BGR2RGB)
            rgb = gpu_rgb.download()
        except Exception:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        res = holistic_model.process(rgb)

        def lm_arr(lm, n):
            return (
                np.array([[p.x, p.y, p.z] for p in lm.landmark], np.float32)
                if lm
                else np.zeros((n, 3), np.float32)
            )

        pose_lm = lm_arr(res.pose_landmarks, 33)
        lh_lm = lm_arr(res.left_hand_landmarks, 21)
        rh_lm = lm_arr(res.right_hand_landmarks, 21)
        face_lm = lm_arr(res.face_landmarks, 468)

        def to_global(pts):
            g = pts.copy()
            g[:, 0] = g[:, 0] * (x2 - x1) + x1
            g[:, 1] = g[:, 1] * (y2 - y1) + y1
            return g

        pose_seq.append(to_global(pose_lm).reshape(-1))
        lh_seq.append(to_global(lh_lm).reshape(-1))
        rh_seq.append(to_global(rh_lm).reshape(-1))
        face_seq.append(to_global(face_lm).reshape(-1))

        if of_feats is not None and idx < len(of_feats):
            head_pose = of_feats[idx, :3]
            aus = of_feats[idx, 3:]
        else:
            head_pose = np.zeros(3, np.float32)
            aus = np.zeros(n_aus, np.float32)
        head_seq.append(head_pose)
        au_seq.append(aus)
        torso_seq.append(_torso_tilt(pose_lm))
        idx += 1

    cap.release()

    flow_seq = compute_optical_flow(path)
    avg_flow = (
        flow_seq.mean(axis=(1, 2)) if flow_seq.size > 0 else np.zeros((len(pose_seq), 2), np.float32)
    )
    if flow_seq.shape[0] + 1 == len(pose_seq):
        avg_flow = np.concatenate([np.zeros((1, 2), np.float32), avg_flow], axis=0)
    mag = np.linalg.norm(avg_flow, axis=-1, keepdims=True)
    flow_node = np.concatenate([avg_flow, mag], axis=1)

    pose_arr = np.stack(pose_seq).reshape(-1, 33, 3)
    lh_arr = np.stack(lh_seq).reshape(-1, 21, 3)
    rh_arr = np.stack(rh_seq).reshape(-1, 21, 3)
    face_arr = np.stack(face_seq).reshape(-1, 468, 3)

    nodes = np.concatenate([pose_arr, lh_arr, rh_arr, face_arr], axis=1)
    if flow_node.shape[0] < nodes.shape[0]:
        flow_node = np.pad(flow_node, ((0, nodes.shape[0] - flow_node.shape[0]), (0, 0)))
    nodes = np.concatenate([nodes, flow_node[:, None, :]], axis=1)

    if of_feats is not None:
        head_arr = np.stack(head_seq).reshape(-1, 1, 3)
        torso_arr = np.stack(torso_seq).reshape(-1, 1, 3)
        nodes = np.concatenate([nodes, head_arr, torso_arr], axis=1)
        if n_aus > 0:
            aus_arr = np.stack(au_seq)
            aus_arr = np.repeat(aus_arr[:, :, None], 3, axis=2)
            nodes = np.concatenate([nodes, aus_arr], axis=1)

    return torch.from_numpy(nodes).permute(2, 0, 1).float().unsqueeze(0)


def pad_batch(feats_list: list[torch.Tensor]) -> torch.Tensor:
    """Pad a list of feature tensors along the time dimension."""
    T = max(f.shape[2] for f in feats_list)
    B = len(feats_list)
    C = feats_list[0].shape[1]
    V = feats_list[0].shape[3]
    batch = torch.zeros(B, C, T, V)
    for i, f in enumerate(feats_list):
        t = f.shape[2]
        batch[i, :, :t] = f[0]
    return batch

