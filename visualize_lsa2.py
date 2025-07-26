#!/usr/bin/env python
# visualize_lsa.py
# Visualizador LSA: landmarks + subtítulos directamente sobre el vídeo,
# con soporte de subtítulos desde columna "splits" o desde columnas individuales,
# con tamaño de ventana configurable y texto Unicode.

import os
import cv2
import argparse
import h5py
import pandas as pd
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import ast

def draw_landmarks(frame, arr3d, connections, color=(0,255,0), radius=2, thickness=1):
    pts = arr3d[:, :2].astype(np.int32)
    for a, b in connections:
        x1, y1 = pts[a]; x2, y2 = pts[b]
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    for x, y in pts:
        cv2.circle(frame, (x, y), radius, color, -1)

# Cargar fuente para Unicode (Pillow)
try:
    FONT = ImageFont.truetype("arial.ttf", 24)
except IOError:
    FONT = ImageFont.load_default()


def visualize(video_dir, h5_path, csv_path, win_size, no_window=False):
    # Leer CSV y HDF5
    df = pd.read_csv(csv_path)
    if 'id' not in df.columns:
        raise ValueError("CSV debe contener columna 'id' con el nombre base de cada vídeo.")
    if 'splits' not in df.columns and not {'video','label','start','end'}.issubset(df.columns):
        raise ValueError("CSV debe contener columna 'splits' o las columnas 'video','label','start','end'.")

    h5f = h5py.File(h5_path, 'r')

    # MediaPipe connections
    mp_holistic  = mp.solutions.holistic
    mp_face_mesh = mp.solutions.face_mesh
    POSE_CONN = mp_holistic.POSE_CONNECTIONS
    HAND_CONN = mp_holistic.HAND_CONNECTIONS
    FACE_CONN = mp_face_mesh.FACEMESH_TESSELATION

    # Configurar ventana
    if not no_window:
        cv2.namedWindow('LSA Visualizer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('LSA Visualizer', win_size[0], win_size[1])

    for vid, grp in h5f.items():
        base, _ = os.path.splitext(vid)
        pose_ds = grp['pose']; lh_ds = grp['left_hand']
        rh_ds   = grp['right_hand']; face_ds = grp['face']

        path = os.path.join(video_dir, vid)
        if not os.path.exists(path): path = os.path.join(video_dir, base + '.mp4')
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"ATENCIÓN: no pude abrir {path}, salto este vídeo.")
            continue

        df_sub = df[df['id'].astype(str) == base]
        if df_sub.empty:
            print(f"No hay subtítulos en CSV para '{base}' (id). Continua sin texto.")
            subs = []
        else:
            row = df_sub.iloc[0]
            if 'splits' in row and pd.notna(row['splits']):
                lst = row['splits']
                splits = ast.literal_eval(lst) if isinstance(lst, str) else lst
                subs = []
                chunk_start = row.get('start', 0.0)
                for text, gstart, gend in splits:
                    local_s = gstart - chunk_start
                    local_e = gend   - chunk_start
                    subs.append({'start_time': local_s, 'end_time': local_e, 'text': str(text)})
            else:
                chunk_start = row['start']; chunk_end = row['end']
                duration    = chunk_end - chunk_start
                subs = [{'start_time': 0.0, 'end_time': duration, 'text': str(row['label'])}]

        print(f"Video {base}: {len(subs)} subtítulos cargados.")

        frame_idx = 0; paused = False
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame_idx >= pose_ds.shape[0]: break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            pose_arr = pose_ds[frame_idx].reshape(-1,3)
            lh_arr   = lh_ds[frame_idx].reshape(-1,3)
            rh_arr   = rh_ds[frame_idx].reshape(-1,3)
            face_arr = face_ds[frame_idx].reshape(-1,3)
            frame_idx += 1

            draw_landmarks(frame, face_arr, FACE_CONN, color=(255,0,0), radius=1)
            draw_landmarks(frame, pose_arr, POSE_CONN, color=(0,255,0), radius=2)
            draw_landmarks(frame, lh_arr, HAND_CONN,   color=(0,0,255), radius=2)
            draw_landmarks(frame, rh_arr, HAND_CONN,   color=(0,0,255), radius=2)
            pts = np.vstack([face_arr[:,:2], pose_arr[:,:2], lh_arr[:,:2], rh_arr[:,:2]]).astype(np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

            text = ""
            for sub in subs:
                if sub['start_time'] <= t <= sub['end_time']:
                    text = sub['text']; break

            if text:
                oy = frame.shape[0] - 30
                pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(pil)
                # Calcular tamaño de texto
                bbox = draw.textbbox((0, 0), text, font=FONT)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.rectangle([10, oy-th-10, 10+tw+10, oy+10], fill=(0,0,0))
                draw.text((15, oy-th), text, font=FONT, fill=(255,255,255))
                frame = np.array(pil)

            if not no_window:
                cv2.imshow('LSA Visualizer', frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            if key == ord('p'): paused = not paused

        cap.release()

    h5f.close()
    if not no_window: cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualizador LSA (landmarks + subtítulos)')
    parser.add_argument('--video_dir', required=True, help='Carpeta de vídeos')
    parser.add_argument('--h5_file',   required=True, help='HDF5 con landmarks')
    parser.add_argument('--csv_file',  required=True, help='CSV con id y splits/label')
    parser.add_argument('--win_width',  type=int, default=1280)
    parser.add_argument('--win_height', type=int, default=720)
    parser.add_argument('--no_window',  action='store_true')
    args = parser.parse_args()

    visualize(
        video_dir = args.video_dir,
        h5_path   = args.h5_file,
        csv_path  = args.csv_file,
        win_size  = (args.win_width, args.win_height),
        no_window = args.no_window
    )
