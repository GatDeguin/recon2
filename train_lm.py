"""Entrena un modelo de lenguaje basado en Transformer.

Este script recibe un corpus de transcripciones, construye un vocabulario
simple por palabras y entrena el modelo ``TransformerLanguageModel`` definido en
``models/transformer_lm.py``. El modelo resultante se guarda en un checkpoint
para su uso posterior en el decodificador del servidor.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.transformer_lm import TransformerLanguageModel


def build_vocab(corpus: Path, vocab_path: Path) -> Dict[str, int]:
    """Build or load a vocabulary file.

    The vocabulary is saved to ``vocab_path`` with one token per line. Special
    tokens ``<unk>``, ``<sos>``, ``<eos>`` and ``<pad>`` occupy the first four
    indices respectively.
    """

    if vocab_path.exists():
        vocab: Dict[str, int] = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                vocab[line.strip()] = i
        return vocab

    specials = ["<unk>", "<sos>", "<eos>", "<pad>"]
    counts: Counter[str] = Counter()
    with open(corpus, "r", encoding="utf-8") as f:
        for line in f:
            counts.update(line.strip().split())

    vocab = {tok: i for i, tok in enumerate(specials)}
    for token, _ in counts.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)

    with open(vocab_path, "w", encoding="utf-8") as f:
        for tok, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{tok}\n")

    return vocab


class TextDataset(Dataset):
    """Dataset simple para entrenamiento de modelos de lenguaje."""

    def __init__(self, corpus: Path, vocab: Dict[str, int], seq_len: int) -> None:
        self.vocab = vocab
        self.seq_len = seq_len
        self.pad_id = vocab["<pad>"]
        self.unk_id = vocab["<unk>"]
        self.sos_id = vocab["<sos>"]
        self.eos_id = vocab["<eos>"]

        self.samples: List[tuple[torch.Tensor, torch.Tensor]] = []
        with open(corpus, "r", encoding="utf-8") as f:
            for line in f:
                ids = [self.sos_id]
                ids += [vocab.get(t, self.unk_id) for t in line.strip().split()]
                ids.append(self.eos_id)

                for i in range(0, len(ids) - 1, seq_len):
                    x = ids[i : i + seq_len]
                    y = ids[i + 1 : i + 1 + seq_len]
                    if len(x) < seq_len:
                        x += [self.pad_id] * (seq_len - len(x))
                        y += [self.pad_id] * (seq_len - len(y))
                    self.samples.append(
                        (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))
                    )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def train_model(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    pad_id: int,
    device: torch.device,
) -> None:
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad()
            out = model(x)
            loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
            loss.backward()
            optim.step()

            total += loss.item()
        avg = total / max(len(loader), 1)
        print(f"epoch {epoch}: loss {avg:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer language model")
    parser.add_argument("corpus", type=Path, help="Archivo de texto con transcripciones")
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("vocab.txt"),
        help="Ruta para el archivo de vocabulario",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/lm.pt"),
        help="Ruta donde guardar el checkpoint del modelo",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    vocab = build_vocab(args.corpus, args.vocab)
    dataset = TextDataset(args.corpus, vocab, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = TransformerLanguageModel(len(vocab))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, loader, args.epochs, args.lr, vocab["<pad>"], device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()

