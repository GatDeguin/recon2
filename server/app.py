from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
import uvicorn
import torch
import io

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

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    video_bytes = io.BytesIO(data)
    # TODO: convertir video_bytes en tensor de features reales
    feats = torch.zeros(1, 3, 1, 1)
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
            # TODO: procesar datos en tensor de features reales
            feats = torch.zeros(1, 3, 1, 1)
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
