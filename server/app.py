import logging
import os
import time
import uuid

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
)
import torch
import uvicorn

from .decoder import decode
from .feature_extraction import extract_features_from_bytes, pad_batch
from .models import (
    GRPC_AVAILABLE,
    device,
    logger as metrics_logger,
    load_models,
    model,
    onnx_sess,
    start_grpc_server,
)


log = logging.getLogger(__name__)

# Validation settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_SEQ_LEN = 1000
ALLOWED_MIME_TYPES = {"video/mp4"}
ALLOWED_EXTENSIONS = {".mp4"}

app = FastAPI()


@app.on_event("startup")
def _load_models() -> None:
    """Load models at application startup."""
    load_models()


@app.on_event("shutdown")
def _close_logger() -> None:
    """Ensure metrics database is closed on shutdown."""
    metrics_logger.close()


@app.post("/transcribe")
async def transcribe(files: list[UploadFile] = File(...)):
    feats_list = []
    video_paths = []
    for file in files:
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported media type")
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported file extension")
        data = await file.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        try:
            start = time.time()
            feats = extract_features_from_bytes(data)
            latency = time.time() - start
        except Exception as e:  # pragma: no cover - safety
            raise HTTPException(
                status_code=400, detail="Corrupt or unsupported video file"
            ) from e
        if feats.shape[2] > MAX_SEQ_LEN:
            raise HTTPException(status_code=400, detail="Video exceeds maximum length")
        feats_list.append(feats)
        fps = feats.shape[2] / latency if latency > 0 else 0.0
        metrics_logger.log(latency=latency, fps=fps)
        uid = uuid.uuid4().hex
        video_path = os.path.join("logs", "videos", f"{uid}.mp4")
        with open(video_path, "wb") as f:
            f.write(data)
        video_paths.append(video_path)

    batch = pad_batch(feats_list)

    transcripts = []
    confs = []
    if onnx_sess is not None:
        outputs = onnx_sess.run(None, {onnx_sess.get_inputs()[0].name: batch.numpy()})[0]
        for logit in outputs:
            t = torch.from_numpy(logit).unsqueeze(0)
            transcripts.append(decode(t))
            confs.append(torch.softmax(t, dim=-1).max(-1).values.mean().item())
    elif model is not None:
        with torch.no_grad():
            logits = model(batch.to(device))
        for logit in logits:
            l = logit.unsqueeze(0).cpu()
            transcripts.append(decode(l))
            confs.append(torch.softmax(l, dim=-1).max(-1).values.mean().item())
    else:
        transcripts = ["" for _ in feats_list]
        confs = [0.0 for _ in feats_list]

    log_path = os.path.join("logs", "predictions.csv")
    header = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as lf:
        if header:
            lf.write("video_path,confidence\n")
        for p, c in zip(video_paths, confs):
            lf.write(f"{p},{c}\n")

    return {"transcripts": transcripts}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            if len(data) > MAX_FILE_SIZE:
                await ws.send_json({"error": "File too large"})
                continue
            try:
                start = time.time()
                feats = extract_features_from_bytes(data)
                latency = time.time() - start
            except Exception:
                await ws.send_json({"error": "Corrupt or unsupported video data"})
                continue
            if feats.shape[2] > MAX_SEQ_LEN:
                await ws.send_json({"error": "Video exceeds maximum length"})
                continue
            fps = feats.shape[2] / latency if latency > 0 else 0.0

            if onnx_sess is not None:
                out = onnx_sess.run(None, {onnx_sess.get_inputs()[0].name: feats.numpy()})[0]
                logits = torch.from_numpy(out)
                transcript = decode(logits)
            elif model is not None:
                with torch.no_grad():
                    logits = model(feats.to(device))
                transcript = decode(logits.cpu())
            else:
                transcript = ""

            metrics_logger.log(latency=latency, fps=fps)
            await ws.send_json({"transcript": transcript})
    except WebSocketDisconnect:
        log.warning("Client disconnected")
        await ws.close()


if __name__ == "__main__":
    load_models()
    if GRPC_AVAILABLE:
        start_grpc_server()
    uvicorn.run(app, host="0.0.0.0", port=8000)

