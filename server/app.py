import os
import time
import uuid

import torch
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
import uvicorn

from .decoder import decode
from .feature_extraction import extract_features_from_bytes, pad_batch
from .models import GRPC_AVAILABLE, device, logger, model, onnx_sess, start_grpc_server


app = FastAPI()


@app.on_event("shutdown")
def _close_logger() -> None:
    """Ensure metrics database is closed on shutdown."""
    logger.close()


@app.post("/transcribe")
async def transcribe(files: list[UploadFile] = File(...)):
    feats_list = []
    video_paths = []
    for file in files:
        data = await file.read()
        uid = uuid.uuid4().hex
        video_path = os.path.join("logs", "videos", f"{uid}.mp4")
        with open(video_path, "wb") as f:
            f.write(data)
        video_paths.append(video_path)

        start = time.time()
        feats_list.append(extract_features_from_bytes(data))
        latency = time.time() - start
        fps = feats_list[-1].shape[2] / latency if latency > 0 else 0.0
        logger.log(latency=latency, fps=fps)

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
            start = time.time()
            feats = extract_features_from_bytes(data)
            latency = time.time() - start
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

            logger.log(latency=latency, fps=fps)
            await ws.send_json({"transcript": transcript})
    except WebSocketDisconnect:
        logger.log(class_acc={"message": "Client disconnected."})
        await ws.close()


if __name__ == "__main__":
    if GRPC_AVAILABLE:
        start_grpc_server()
    uvicorn.run(app, host="0.0.0.0", port=8000)

