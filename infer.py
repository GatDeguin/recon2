import argparse
from pathlib import Path
import os
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch import nn

from models.transformer_lm import TransformerLanguageModel, load_model


def load_features(video_path: str, h5_path: str) -> torch.Tensor:
    """Carga las features del HDF5 asociado al video."""
    base = os.path.basename(video_path)
    with h5py.File(h5_path, "r") as h5f:
        key = base if base in h5f else os.path.splitext(base)[0] + ".mp4"
        grp = h5f[key]
        pose = grp["pose"][()]
        lh = grp["left_hand"][()]
        rh = grp["right_hand"][()]
        face = grp["face"][()]
        flow = grp["optical_flow"][()]
    feats = np.concatenate(
        [
            pose.reshape(pose.shape[0], -1),
            lh.reshape(lh.shape[0], -1),
            rh.reshape(rh.shape[0], -1),
            face.reshape(face.shape[0], -1),
            flow.reshape(flow.shape[0], -1),
        ],
        axis=1,
    )
    return torch.from_numpy(feats).float().unsqueeze(0)


def beam_search(
    logits: torch.Tensor,
    lm: TransformerLanguageModel,
    beam: int = 5,
    lm_weight: float = 0.5,
    sos: int = 1,
    eos: int = 2,
) -> List[int]:
    """Beam-search autoregresivo sencillo."""
    vocab_size = logits.size(-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    beams = [([sos], 0.0)]
    for t in range(logits.size(1)):
        candidates = []
        for seq, score in beams:
            if seq[-1] == eos:
                candidates.append((seq, score))
                continue
            for v in range(vocab_size):
                new_seq = seq + [v]
                new_score = score + log_probs[0, t, v].item()
                if lm is not None:
                    inp = torch.tensor(new_seq).unsqueeze(0)
                    with torch.no_grad():
                        lm_logits = lm(inp)
                    lm_lp = torch.log_softmax(lm_logits[0, -1], dim=-1)[v].item()
                    new_score += lm_weight * lm_lp
                candidates.append((new_seq, new_score))
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam]
    return beams[0][0]


def load_vocab(path: str) -> Dict[int, str]:
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            token = line.strip()
            mapping[i] = token
    return mapping


def main() -> None:
    p = argparse.ArgumentParser(description="Inferencia con beam-search")
    p.add_argument("--checkpoint", help="Modelo de reconocimiento")
    p.add_argument("--lm", help="Checkpoint LM")
    p.add_argument("--vocab", help="Archivo vocabulario (uno por línea)")
    p.add_argument("--video", help="Video de entrada")
    p.add_argument("--h5", help="HDF5 con features")
    p.add_argument("--beam", type=int, default=5, help="Tamaño de beam")
    p.add_argument("--lm_weight", type=float, default=0.5, help="Peso del LM")
    args = p.parse_args()

    cfg_path = Path(__file__).resolve().parent / "configs" / "config.yaml"
    from utils.config import load_config, apply_defaults

    cfg = load_config(cfg_path)
    apply_defaults(args, cfg)

    vocab = load_vocab(args.vocab)
    feat = load_features(args.video, args.h5)

    model = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(model, nn.Module):
        model.eval()
    lm = load_model(args.lm, vocab_size=len(vocab))

    with torch.no_grad():
        logits = model(feat)
    tokens = beam_search(logits, lm, args.beam, args.lm_weight)

    words = [vocab.get(t, "<unk>") for t in tokens if t not in (1, 2)]
    print("Transcripci\u00f3n:", " ".join(words))


if __name__ == "__main__":
    main()
