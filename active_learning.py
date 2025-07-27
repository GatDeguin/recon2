#!/usr/bin/env python
"""Selecciona ejemplos de baja confianza para reentrenamiento."""
import os
import argparse
import h5py
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple
import shutil
import pandas as pd

from train import SignDataset, collate, build_model


def compute_scores(model: torch.nn.Module, loader: DataLoader) -> List[Tuple[str, float]]:
    """Compute average confidence for each sample in the loader."""
    device = next(model.parameters()).device
    scores: List[Tuple[str, float]] = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            # collate returns eight values; we only need the features
            feats = batch[0].to(device)

            out = model(feats)
            if isinstance(out, (tuple, list)):
                gloss_logits = out[0]
            else:
                gloss_logits = out

            conf = gloss_logits.exp().max(-1).values.mean().item()
            vid = loader.dataset.samples[idx][0]
            scores.append((vid, conf))
    return scores


def save_selected(scores: List[Tuple[str, float]], dataset: SignDataset, out_dir: str, top_k: int = 10) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_h5 = h5py.File(os.path.join(out_dir, "selected.h5"), "w")
    csv_lines = []
    for vid, _ in sorted(scores, key=lambda x: x[1])[:top_k]:
        lbl = next(s[1] for s in dataset.samples if s[0] == vid)
        grp = dataset.h5[vid]
        new_grp = out_h5.create_group(vid)
        for k in grp.keys():
            new_grp.create_dataset(k, data=grp[k][()])
        csv_lines.append(f"{os.path.splitext(vid)[0]};{lbl}")
    with open(os.path.join(out_dir, "selected.csv"), "w", encoding="utf-8") as f:
        for line in csv_lines:
            f.write(line + "\n")
    out_h5.close()


def load_production_samples(log_file: str, top_k: int) -> List[str]:
    """Return video paths with lowest confidence from production logs."""
    df = pd.read_csv(log_file)
    df = df.sort_values("confidence").head(top_k)
    return df["video_path"].tolist()


def copy_videos(paths: List[str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for pth in paths:
        if os.path.exists(pth):
            shutil.copy2(pth, os.path.join(out_dir, os.path.basename(pth)))


def main() -> None:
    p = argparse.ArgumentParser(description="Proceso simple de active learning")
    p.add_argument("--checkpoint", required=True, help="Modelo entrenado")
    p.add_argument("--h5_file", required=True)
    p.add_argument("--csv_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--top_k", type=int, default=10, help="Cantidad de ejemplos a seleccionar")
    p.add_argument("--log_file", help="CSV con registros de producción")
    args = p.parse_args()

    with SignDataset(args.h5_file, args.csv_file) as ds:
        dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(ckpt, dict):
            model = build_model("stgcn", len(ds.vocab))
            model.load_state_dict(ckpt["model_state"])
        else:
            model = ckpt
        model.eval()

        if args.log_file and os.path.exists(args.log_file):
            paths = load_production_samples(args.log_file, args.top_k)
            copy_videos(paths, os.path.join(args.out_dir, "videos"))
        else:
            scores = compute_scores(model, dl)
            save_selected(scores, ds, args.out_dir, args.top_k)


if __name__ == "__main__":
    main()
