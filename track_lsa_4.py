#!/usr/bin/env python
# track_lsa_multi_signer_gpu_updated.py
# Pipeline LSA multi‑signer con aceleración GPU para YOLOv8 y preprocesado con OpenCV CUDA.
# Detecta límites de seña mediante PELT (ruptures) o variaciones de velocidad y
# guarda cada segmento por separado en el fichero HDF5.

import os
import cv2
import argparse
import numpy as np
import h5py
from tqdm import tqdm
import mediapipe as mp
from ultralytics import YOLO
from optical_flow.raft_runner import compute_optical_flow
import torch

try:
    import ruptures as rpt  # optional dependency
except Exception:  # pragma: no cover - optional dependency
    rpt = None

try:
    import onnxruntime as ort  # optional, only when using YOLOX
except Exception:  # pragma: no cover - optional dependency
    ort = None


def box_iou(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def norm(p, q):
    return np.hypot(p[0] - q[0], p[1] - q[1])


def infer_active_signer(prev_centers, cur_centers,
                        motion_ema, alpha=0.3, MARGIN=1.2,
                        SWITCH_DWELL=5, dwell_state=None):
    # Actualiza EMA y decide si cambiar de signer activo
    for id_, (pl, pr) in prev_centers.items():
        if id_ in cur_centers:
            cl, cr = cur_centers[id_]
            d = norm(pl, cl) + norm(pr, cr)
            prev_ema = motion_ema.get(id_, d)
            motion_ema[id_] = prev_ema * (1 - alpha) + d * alpha
    candidate = max(motion_ema, key=motion_ema.get) if motion_ema else None
    if dwell_state is None:
        dwell_state = {'active_id': candidate, 'count': 0}
    active_id = dwell_state['active_id']
    count = dwell_state['count']
    if active_id is None:
        active_id = candidate; count = 0
    elif candidate != active_id and candidate is not None:
        if motion_ema.get(candidate, 0) > motion_ema.get(active_id, 0) * MARGIN:
            count += 1
        else:
            count = 0
        if count >= SWITCH_DWELL:
            active_id = candidate; count = 0
    else:
        count = 0
    dwell_state['active_id'] = active_id
    dwell_state['count'] = count
    return active_id, dwell_state


def _segment_sequences(lh_seq, rh_seq, face_seq, penalty=10):
    """Detect change points using PELT on landmark velocities."""
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
    else:  # Fallback using simple threshold on speed
        thr = speed.mean() * 0.5
        boundaries = [0]
        for i in range(1, len(speed)):
            if speed[i] < thr and speed[max(0, i-1)] >= thr:
                boundaries.append(i)
        boundaries.append(len(speed))

    segments = []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if e > s:
            segments.append((s, e))
    if not segments:
        segments.append((0, len(speed)))
    return segments


def procesar_videos(input_dir, output_file,
                    yolo_conf=0.5, mp_conf=0.7,
                    show_window=True, yolox_model=None):
    """Procesa los vídeos de *input_dir* almacenando landmarks en *output_file*.

    Si *yolox_model* es una ruta a un modelo YOLOX exportado a ONNX se
    utilizará dicho modelo para la detección de personas; de lo contrario se
    empleará YOLOv8 desde ``ultralytics``.
    """

    yolox_sess = None
    if yolox_model:
        if ort is None:
            raise RuntimeError("onnxruntime no está disponible")
        providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        yolox_sess = ort.InferenceSession(yolox_model, providers=providers)
    else:
        # Inicializar YOLOv8 sin args en constructor
        yolo_model = YOLO('yolov8n.pt')
    mp_holistic  = mp.solutions.holistic
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing   = mp.solutions.drawing_utils

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=mp_conf,
        min_tracking_confidence=mp_conf
    )

    with h5py.File(output_file, 'w') as h5f:
        videos = sorted([f for f in os.listdir(input_dir)
                         if f.lower().endswith(('.mp4','.avi','.mov'))])
        for vid in videos:
            cap = cv2.VideoCapture(os.path.join(input_dir, vid))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pose_seq, lh_seq, rh_seq, face_seq = [], [], [], []
            prev_boxes, prev_ids = [], []
            prev_centers, motion_ema = {}, {}
            dwell_state = {'active_id': None, 'count': 0}
            next_id = 0
            pbar = tqdm(total=total, desc=f"[{vid}]", unit='fr',
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            pbar.set_postfix(yolo_conf=yolo_conf, mp_conf=mp_conf)

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
                    valid = preds[(preds[:, 4] >= yolo_conf) & (preds[:, 5] == 0)]
                    scale_x = frame.shape[1] / iw
                    scale_y = frame.shape[0] / ih
                    boxes = valid[:, :4].copy()
                    boxes[:, 0] *= scale_x
                    boxes[:, 2] *= scale_x
                    boxes[:, 1] *= scale_y
                    boxes[:, 3] *= scale_y
                    boxes = boxes.astype(int)
                else:
                    yres = yolo_model(frame, device=0, half=True, conf=yolo_conf, classes=[0])[0]
                    boxes = yres.boxes.xyxy.cpu().numpy().astype(int)

                # Asignar IDs persistentes
                cur_ids = [-1]*len(boxes)
                cur_boxes = [tuple(b) for b in boxes]
                for i, cb in enumerate(cur_boxes):
                    best_iou, best_j = 0, -1
                    for j, pb in enumerate(prev_boxes):
                        iou = box_iou(cb, pb)
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    if best_iou > 0.3:
                        cur_ids[i] = prev_ids[best_j]
                    else:
                        cur_ids[i] = next_id; next_id += 1

                signer_data, cur_centers = [], {}
                for box, id_ in zip(cur_boxes, cur_ids):
                    x1,y1,x2,y2 = box
                    roi = frame[y1:y2, x1:x2]
                    # Preprocesado GPU con OpenCV CUDA
                    try:
                        gpu_roi = cv2.cuda_GpuMat(); gpu_roi.upload(roi)
                        gpu_rgb = cv2.cuda.cvtColor(gpu_roi, cv2.COLOR_BGR2RGB)
                        rgb = gpu_rgb.download()
                    except Exception:
                        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                    res = holistic.process(rgb)
                    # Dibujar landmarks en ROI...
                    if res.face_landmarks:
                        mp_drawing.draw_landmarks(roi, res.face_landmarks,
                            mp_face_mesh.FACEMESH_TESSELATION,
                            mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(thickness=1))
                    if res.pose_landmarks:
                        mp_drawing.draw_landmarks(roi, res.pose_landmarks,
                            mp_holistic.POSE_CONNECTIONS)
                    if res.left_hand_landmarks:
                        mp_drawing.draw_landmarks(roi, res.left_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS)
                    if res.right_hand_landmarks:
                        mp_drawing.draw_landmarks(roi, res.right_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS)

                    # Extraer arrays de landmarks...
                    def lm_arr(lm, n):
                        return np.array([[p.x,p.y,p.z] for p in lm.landmark], np.float32) \
                            if lm else np.zeros((n,3), np.float32)

                    pose_lm = lm_arr(res.pose_landmarks,33)
                    lh_lm   = lm_arr(res.left_hand_landmarks,21)
                    rh_lm   = lm_arr(res.right_hand_landmarks,21)
                    face_lm = lm_arr(res.face_landmarks,468)

                    def to_global(pts):
                        g=pts.copy(); g[:,0]=g[:,0]*(x2-x1)+x1; g[:,1]=g[:,1]*(y2-y1)+y1; return g

                    pose_g = to_global(pose_lm).reshape(-1)
                    lh_g   = to_global(lh_lm).reshape(-1)
                    rh_g   = to_global(rh_lm).reshape(-1)
                    face_g = to_global(face_lm).reshape(-1)
                    lh_c = tuple(to_global(lh_lm)[:,:2].mean(axis=0))
                    rh_c = tuple(to_global(rh_lm)[:,:2].mean(axis=0))

                    signer_data.append({'id':id_,'box':box,
                                        'pose':pose_g,'lh':lh_g,
                                        'rh':rh_g,'face':face_g})
                    cur_centers[id_] = (lh_c,rh_c)
                    frame[y1:y2,x1:x2] = roi

                # Inferir signer activo con EMA + histéresis
                active_id,dwell_state = infer_active_signer(
                    prev_centers,cur_centers,motion_ema,
                    alpha=0.3,MARGIN=1.2,SWITCH_DWELL=5,dwell_state=dwell_state)

                # Acumular landmarks y dibujar bbox
                for sd in signer_data:
                    pose_seq.append(sd['pose'])
                    lh_seq.append(sd['lh'])
                    rh_seq.append(sd['rh'])
                    face_seq.append(sd['face'])
                    x1,y1,x2,y2 = sd['box']
                    color = (0,255,0) if sd['id']==active_id else (255,255,255)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    if sd['id']==active_id:
                        cv2.putText(frame,f"Signer {active_id}",(x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

                if show_window:
                    cv2.imshow('LSA Tracking GPU',frame)
                    if cv2.waitKey(1)&0xFF==ord('q'): break
                pbar.update(1)
                prev_boxes,prev_ids,prev_centers = cur_boxes,cur_ids,cur_centers.copy()

            pbar.close(); cap.release()
            flow_seq = compute_optical_flow(os.path.join(input_dir, vid))

            segments = _segment_sequences(lh_seq, rh_seq, face_seq)
            pose_arr = np.stack(pose_seq)
            lh_arr = np.stack(lh_seq)
            rh_arr = np.stack(rh_seq)
            face_arr = np.stack(face_seq)

            grp = h5f.create_group(vid)
            for i, (s, e) in enumerate(segments):
                sg = grp.create_group(f"segment_{i:03d}")
                sg.create_dataset('pose', data=pose_arr[s:e], compression='gzip')
                sg.create_dataset('left_hand', data=lh_arr[s:e], compression='gzip')
                sg.create_dataset('right_hand', data=rh_arr[s:e], compression='gzip')
                sg.create_dataset('face', data=face_arr[s:e], compression='gzip')
                if flow_seq.size > 0 and e - s > 1:
                    sg.create_dataset('optical_flow', data=flow_seq[s:e-1], compression='gzip')
                else:
                    sg.create_dataset('optical_flow', data=np.empty((0,), np.float32), compression='gzip')

    holistic.close()
    if show_window: cv2.destroyAllWindows()
    print(f"\n✅ Datos volcados en {output_file}")

if __name__=='__main__':
    parser=argparse.ArgumentParser(description="LSA GPU: Tracking multi‐signer → HDF5")
    parser.add_argument('--input_dir',required=True,help="Carpeta con vídeos")
    parser.add_argument('--output_file',required=True,help="Salida .h5")
    parser.add_argument('--yolo_conf',type=float,default=0.5,help="Conf YOLO")
    parser.add_argument('--mp_conf',type=float,default=0.7,help="Conf MediaPipe")
    parser.add_argument('--no_window',action='store_true',help="Ocultar ventana de visualización")
    parser.add_argument('--yolox_model',help="Ruta a modelo YOLOX ONNX")
    args=parser.parse_args()
    procesar_videos(args.input_dir,args.output_file,
                    args.yolo_conf,args.mp_conf,
                    not args.no_window,
                    args.yolox_model)
