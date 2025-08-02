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
from optical_flow.raft_runner import compute_optical_flow
import torch
import pandas as pd
import subprocess
import tempfile
import shutil

from utils.pipeline import PersonDetector, LandmarkExtractor, segment_sequences

OPENFACE_BIN = os.environ.get("OPENFACE_BIN")


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


def _run_openface(video_path: str):
    """Return array of [Rx,Ry,Rz,AUs] per frame or None if disabled."""
    if not OPENFACE_BIN:
        return None
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, "of.csv")
    cmd = [OPENFACE_BIN, "-f", video_path, "-aus", "-pose", "-of", csv_path, "-q"]
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


def procesar_videos(input_dir, output_file,
                    yolo_conf=0.5, mp_conf=0.7,
                    show_window=True, yolox_model=None):
    """Procesa los vídeos de *input_dir* almacenando landmarks en *output_file*.

    Si *yolox_model* es una ruta a un modelo YOLOX exportado a ONNX se
    utilizará dicho modelo para la detección de personas; de lo contrario se
    empleará YOLOv8 desde ``ultralytics``.
    """

    detector = PersonDetector(conf=yolo_conf, yolox_model=yolox_model)
    extractor = LandmarkExtractor(mp_conf=mp_conf)

    with h5py.File(output_file, 'w') as h5f:
        videos = sorted([f for f in os.listdir(input_dir)
                         if f.lower().endswith(('.mp4','.avi','.mov'))])
        for vid in videos:
            cap = cv2.VideoCapture(os.path.join(input_dir, vid))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            of_feats = _run_openface(os.path.join(input_dir, vid))
            n_aus = of_feats.shape[1] - 3 if of_feats is not None else 0
            pose_seq, lh_seq, rh_seq, face_seq = [], [], [], []
            head_seq, torso_seq, au_seq = [], [], []
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
                boxes = detector.detect(frame)

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
                    data = extractor.extract(frame, box)
                    signer_data.append({'id': id_, 'box': box,
                                        'pose': data['pose'],
                                        'lh': data['left_hand'],
                                        'rh': data['right_hand'],
                                        'face': data['face']})
                    cur_centers[id_] = (data['lh_center'], data['rh_center'])

                pose_lm = signer_data[0]['pose'].reshape(33, 3) if signer_data else np.zeros((33, 3), np.float32)

                idx = len(pose_seq)
                if of_feats is not None and idx < len(of_feats):
                    head_pose = of_feats[idx,:3]
                    aus = of_feats[idx,3:]
                else:
                    head_pose = np.zeros(3, np.float32)
                    aus = np.zeros(n_aus, np.float32)
                head_seq.append(head_pose)
                au_seq.append(aus)
                torso_seq.append(_torso_tilt(pose_lm))

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

            segments = segment_sequences(lh_seq, rh_seq, face_seq)
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
                if of_feats is not None:
                    sg.create_dataset('head_pose', data=np.stack(head_seq)[s:e], compression='gzip')
                    sg.create_dataset('torso_pose', data=np.stack(torso_seq)[s:e], compression='gzip')
                    sg.create_dataset('aus', data=np.stack(au_seq)[s:e], compression='gzip')

    extractor.close()
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
