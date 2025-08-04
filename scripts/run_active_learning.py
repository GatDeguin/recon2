#!/usr/bin/env python
"""Wrapper to extract low-confidence samples for re-annotation.

This script finds the most recent checkpoint, scores the dataset using
``active_learning.compute_scores`` and copies the least confident examples
(with their videos and feature snippets) to a folder or cloud bucket.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from active_learning import (
    compute_scores,
    save_selected,
    copy_videos,
    load_production_samples,
)
from train import SignDataset, collate, build_model
from utils.config import load_config, apply_defaults


_DEF_TOP_K = 10


def _latest_checkpoint(ckpt_dir: str | Path) -> Path:
    """Return newest ``.pt/.pth`` file inside ``ckpt_dir``."""

    ckpt_dir = Path(ckpt_dir)
    candidates = list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _is_remote(path: str) -> bool:
    return path.startswith("gs://") or path.startswith("s3://")


def _upload_dir(local: str, remote: str) -> None:
    """Upload *local* directory to *remote* bucket using gsutil or aws CLI."""

    if remote.startswith("gs://"):
        subprocess.run(["gsutil", "cp", "-r", local, remote], check=True)
    elif remote.startswith("s3://"):
        subprocess.run(["aws", "s3", "cp", "--recursive", local, remote], check=True)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported remote destination: {remote}")


def main() -> None:
    p = argparse.ArgumentParser(description="Select samples for active learning")
    p.add_argument("--h5_file")
    p.add_argument("--csv_file")
    p.add_argument("--checkpoint_dir")
    p.add_argument("--out_dir", default="reannotation")
    p.add_argument("--top_k", type=int, default=_DEF_TOP_K)
    p.add_argument("--video_dir", help="Folder with original videos to copy")
    p.add_argument("--log_file", help="CSV with production scores")
    p.add_argument("--arch", help="Model architecture")
    args = p.parse_args()

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    cfg = load_config(cfg_path)
    apply_defaults(args, cfg)

    ckpt_path = _latest_checkpoint(args.checkpoint_dir)

    out_dir = args.out_dir
    if _is_remote(out_dir):
        tmp_dir = tempfile.mkdtemp(prefix="reannot_")
        local_out = tmp_dir
    else:
        local_out = out_dir

    with SignDataset(args.h5_file, args.csv_file) as ds:
        dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            model = build_model(args.arch, len(ds.vocab))
            model.load_state_dict(ckpt["model_state"])
        else:
            model = ckpt
        model.eval()

        if args.log_file and os.path.exists(args.log_file):
            paths = load_production_samples(args.log_file, args.top_k)
            copy_videos(paths, os.path.join(local_out, "videos"))
        else:
            scores = compute_scores(model, dl)
            save_selected(scores, ds, local_out, args.top_k)
            if args.video_dir:
                vids = [v for v, _ in sorted(scores, key=lambda x: x[1])[: args.top_k]]
                paths = [os.path.join(args.video_dir, v) for v in vids]
                copy_videos(paths, os.path.join(local_out, "videos"))

    if _is_remote(out_dir):
        _upload_dir(local_out, out_dir)
        shutil.rmtree(local_out, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
