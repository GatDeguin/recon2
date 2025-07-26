import torch
import cv2
import numpy as np

_model = None
_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _load_model():
    global _model
    if _model is None:
        _model = torch.hub.load('princeton-vl/RAFT', 'raft', pretrained=True)
        _model = _model.to(_device)
        _model.eval()
    return _model

def compute_optical_flow(video_path):
    """Compute dense optical flow sequence for a video using RAFT."""
    model = _load_model()
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return np.empty((0,), np.float32)
    prev_t = torch.from_numpy(prev).permute(2,0,1).float() / 255.0
    prev_t = prev_t.unsqueeze(0).to(_device)
    flows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur_t = torch.from_numpy(frame).permute(2,0,1).float() / 255.0
        cur_t = cur_t.unsqueeze(0).to(_device)
        with torch.no_grad():
            _, flow_up = model(prev_t, cur_t, iters=20, test_mode=True)
        flow_np = flow_up[0].permute(1,2,0).cpu().numpy().astype(np.float32)
        flows.append(flow_np)
        prev_t = cur_t
    cap.release()
    return np.stack(flows) if flows else np.empty((0,), np.float32)
