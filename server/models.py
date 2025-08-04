import os
import torch
import mediapipe as mp
from ultralytics import YOLO
from metrics import MetricsLogger

try:  # optional dependency for YOLOX and ONNX backend
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

# Optional OpenFace binary
OPENFACE_BIN = os.environ.get("OPENFACE_BIN")

# Initialize detection models
mp_holistic = mp.solutions.holistic
yolox_path = os.environ.get("YOLOX_ONNX")
if yolox_path and ort is not None:
    providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    yolox_sess = ort.InferenceSession(yolox_path, providers=providers)
    yolo_model = None
else:
    yolox_sess = None
    yolo_model = YOLO("yolov8n.pt")

holistic_model = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Logging
os.makedirs("logs/videos", exist_ok=True)
logger = MetricsLogger(os.path.join("logs", "metrics.db"))

# Backend selection: cpu, cuda or onnx
BACKEND = os.environ.get("BACKEND", "cpu").lower()
ONNX_PATH = os.environ.get("ONNX_MODEL", "checkpoints/model.onnx")

model = None
onnx_sess = None
device = "cpu"

if BACKEND == "onnx":
    if ort is not None:
        providers = ["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"]
        try:
            onnx_sess = ort.InferenceSession(ONNX_PATH, providers=providers)
        except Exception:
            onnx_sess = None
    else:
        onnx_sess = None
else:
    device = "cuda" if BACKEND == "cuda" and torch.cuda.is_available() else "cpu"
    try:
        model = torch.jit.load("checkpoints/model.ts", map_location=device)
        model.eval()
    except Exception:
        model = None

# Optional gRPC support
try:
    import grpc  # type: ignore
    GRPC_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    GRPC_AVAILABLE = False
    grpc = None


def start_grpc_server(port: int = 50051):
    """Start a gRPC server if grpc is available."""
    if not GRPC_AVAILABLE:
        return None

    import threading
    from concurrent import futures
    from server.protos import transcriber_pb2, transcriber_pb2_grpc
    from .feature_extraction import extract_features_from_bytes
    from .decoder import decode

    class TranscriberService(transcriber_pb2_grpc.TranscriberServicer):
        def Transcribe(self, request, context):
            feats = extract_features_from_bytes(request.video)
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
            return transcriber_pb2.TranscriptReply(transcript=transcript)

    def _serve() -> None:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        transcriber_pb2_grpc.add_TranscriberServicer_to_server(TranscriberService(), server)
        server.add_insecure_port(f"[::]:{port}")
        server.start()
        server.wait_for_termination()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t
