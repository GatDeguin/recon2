from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
import uvicorn
import torch
import tempfile
import numpy as np
import cv2
import mediapipe as mp
from ultralytics import YOLO
from optical_flow.raft_runner import compute_optical_flow

app = FastAPI()

# Cargar modelo TorchScript u ONNX
try:
    model = torch.jit.load("checkpoints/model.ts")
    model.eval()
except Exception:
    model = None

# Cargar vocabulario si existe
vocab = {}
try:
    with open("vocab.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            vocab[i] = line.strip()
except FileNotFoundError:
    pass

SOS_TOKEN = 1
EOS_TOKEN = 2

def decode(logits: torch.Tensor) -> str:
    if logits.dim() == 3:
        tokens = logits.argmax(dim=-1)[0]
    else:
        tokens = logits.argmax(dim=-1)
    words = [vocab.get(int(t), "<unk>") for t in tokens if int(t) not in (SOS_TOKEN, EOS_TOKEN)]
    return " ".join(words)


def extract_features_from_bytes(data: bytes) -> torch.Tensor:
    """Generate ST-GCN features from raw video bytes."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        return _extract_features(tmp.name)


def _extract_features(path: str) -> torch.Tensor:
    """Compute landmarks and optical flow for a video path."""
    yolo = YOLO("yolov8n.pt")
    mp_holistic = mp.solutions.holistic

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(path)
    pose_seq, lh_seq, rh_seq, face_seq = [], [], [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yres = yolo(frame, device=0 if torch.cuda.is_available() else "cpu", half=torch.cuda.is_available(), conf=0.5, classes=[0])[0]
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
            gpu_roi = cv2.cuda_GpuMat(); gpu_roi.upload(roi)
            gpu_rgb = cv2.cuda.cvtColor(gpu_roi, cv2.COLOR_BGR2RGB)
            rgb = gpu_rgb.download()
        except Exception:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        res = holistic.process(rgb)

        def lm_arr(lm, n):
            return np.array([[p.x, p.y, p.z] for p in lm.landmark], np.float32) if lm else np.zeros((n, 3), np.float32)

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

    cap.release()
    holistic.close()

    flow_seq = compute_optical_flow(path)
    avg_flow = flow_seq.mean(axis=(1, 2)) if flow_seq.size > 0 else np.zeros((len(pose_seq), 2), np.float32)
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

    return torch.from_numpy(nodes).permute(2, 0, 1).float().unsqueeze(0)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    feats = extract_features_from_bytes(data)
    if model is None:
        return {"transcript": ""}
    with torch.no_grad():
        logits = model(feats)
    transcript = decode(logits)
    return {"transcript": transcript}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            feats = extract_features_from_bytes(data)
            if model is None:
                await ws.send_json({"transcript": ""})
                continue
            with torch.no_grad():
                logits = model(feats)
            transcript = decode(logits)
            await ws.send_json({"transcript": transcript})
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
