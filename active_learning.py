#!/usr/bin/env python
"""Selecciona ejemplos de baja confianza para reentrenamiento."""
import os
import argparse
from pathlib import Path
import h5py
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple
import shutil
import pandas as pd

from train import SignDataset, collate, build_model


def compute_scores(model: torch.nn.Module, loader: DataLoader) -> List[Tuple[str, float]]:
    """Compute average confidence for each sample in ``loader``.

    The function works with any batch size and returns scores in the same
    order as ``loader.dataset.samples`` regardless of the loader's batching
    strategy.
    """

    device = next(model.parameters()).device
    scores: List[Tuple[str, float]] = [None] * len(loader.dataset)
    model.eval()
    with torch.no_grad():
        for batch_idx, (feats, *_rest) in enumerate(loader):
            feats = feats.to(device)

            out = model(feats)
            if isinstance(out, (tuple, list)):
                gloss_logits = out[0]
            else:
                gloss_logits = out

            conf_batch = gloss_logits.exp().max(-1).values.mean(dim=1)
            for i, conf in enumerate(conf_batch.tolist()):
                dataset_idx = batch_idx * loader.batch_size + i
                vid = loader.dataset.samples[dataset_idx][0]
                scores[dataset_idx] = (vid, float(conf))

    return [s for s in scores if s is not None]


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
    p.add_argument("--checkpoint")
    p.add_argument("--h5_file")
    p.add_argument("--csv_file")
    p.add_argument("--out_dir")
    p.add_argument("--top_k", type=int)
    p.add_argument("--log_file", help="CSV con registros de producción")
    args = p.parse_args()

    cfg_path = Path(__file__).resolve().parent / "configs" / "config.yaml"
    from utils.config import load_config, apply_defaults

    cfg = load_config(cfg_path)
    apply_defaults(args, cfg)

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
