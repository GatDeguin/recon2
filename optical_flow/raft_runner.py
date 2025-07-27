import os
from pathlib import Path

import torch
import cv2
import numpy as np

_model = None
_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _load_model():
    global _model
    if _model is None:
        ckpt = os.environ.get("RAFT_CKPT")
        if ckpt is None:
            default_path = Path(__file__).resolve().parent.parent / "checkpoints" / "raft-sintel.pth"
            if default_path.exists():
                ckpt = str(default_path)

        repo_env = os.environ.get("RAFT_REPO")
        if repo_env:
            repo_dir = Path(repo_env)
            if not repo_dir.exists():
                raise FileNotFoundError(f"RAFT_REPO path does not exist: {repo_env}")
            repo_or_dir = str(repo_dir)
        else:
            # Attempt to locate RAFT in the torch hub cache
            hub_dir = Path(torch.hub.get_dir())
            cached = list(hub_dir.glob("princeton-vl_RAFT*"))
            if not cached:
                raise RuntimeError(
                    "RAFT repository not found in torch hub cache. "
                    "Set RAFT_REPO to a local clone of the RAFT repository."
                )
            repo_or_dir = str(cached[0])

        if ckpt is not None:
            _model = torch.hub.load(repo_or_dir, 'raft', pretrained=False, ckpt=ckpt, source='local')
        else:
            _model = torch.hub.load(repo_or_dir, 'raft', pretrained=True, source='local')

        _model = _model.to(_device)
        _model.eval()
    return _model

def compute_optical_flow(video_path):
    """Compute dense optical flow sequence for a video using RAFT."""
    try:
        model = _load_model()
    except RuntimeError as e:
        if "RAFT repository not found" in str(e):
            raise RuntimeError(
                "RAFT repository not found. Set the RAFT_REPO environment variable "
                "to a local clone of https://github.com/princeton-vl/RAFT or "
                "download 'raft-sintel.pth' from the RAFT releases page and set "
                "RAFT_CKPT to its location."
            ) from e
        raise
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
