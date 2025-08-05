#!/usr/bin/env python
"""Compute WER and CER on a random validation subset using evaluate.py."""
import argparse
import json
import random
from functools import partial

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from train import SignDataset, collate, build_model  # noqa: E402
from evaluate import compute_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WER/CER on a subset")
    parser.add_argument("--h5_file", required=True, help="HDF5 file with landmarks")
    parser.add_argument("--csv_file", required=True, help="CSV file with labels")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument(
        "--model",
        choices=["stgcn", "sttn", "corrnet+", "mcst"],
        default="stgcn",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--subset",
        type=int,
        default=100,
        help="Number of validation samples to evaluate",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--segments", action="store_true", help="Use segmented samples")
    parser.add_argument(
        "--include_openface", action="store_true", help="Include extra OpenFace features"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    with SignDataset(
        args.h5_file,
        args.csv_file,
        segments=args.segments,
        include_openface=args.include_openface,
    ) as ds:
        idx = list(range(len(ds)))
        random.shuffle(idx)
        idx = idx[: args.subset]
        subset = Subset(ds, idx)
        dl = DataLoader(subset, batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(
            args.model,
            len(ds.vocab),
            len(ds.nmm_vocab),
            len(ds.suffix_vocab),
            len(ds.rnm_vocab),
            len(ds.person_vocab),
            len(ds.number_vocab),
            len(ds.tense_vocab),
            len(ds.aspect_vocab),
            len(ds.mode_vocab),
            num_nodes=ds.num_nodes,
        )
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        inv_vocab = {v: k for k, v in ds.vocab.items()}
        wer, cer, _ = compute_metrics(model, dl, inv_vocab, device)
        print(json.dumps({"wer": wer, "cer": cer}, indent=2))


if __name__ == "__main__":
    main()
