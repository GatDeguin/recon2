#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import SignDataset, collate, build_model, evaluate
from models.stgcn import STGCNBlock
from models.sttn import STTN
from models.corrnet import CorrNetPlus
from models.mcst_transformer import MCSTTransformer


class STGCNStudent(nn.Module):
    """Reduced version of STGCN for knowledge distillation."""

    def __init__(
        self,
        in_channels: int,
        num_class: int,
        num_nodes: int,
        num_nmm: int = 0,
        num_suffix: int = 0,
        num_rnm: int = 0,
    ) -> None:
        super().__init__()
        cfg_path = Path(__file__).resolve().parent / "configs" / "skeleton.yaml"
        if cfg_path.exists():
            try:
                from utils.build_adjacency import build_adjacency

                A = build_adjacency(cfg_path)
            except Exception:
                A = torch.eye(num_nodes)
        else:
            A = torch.eye(num_nodes)
        self.data_bn = nn.BatchNorm1d(num_nodes * in_channels)
        self.layer1 = STGCNBlock(in_channels, 32, A)
        self.layer2 = STGCNBlock(32, 32, A)
        self.layer3 = STGCNBlock(32, 64, A)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.ctc_head = nn.Linear(64, num_class)
        self.nmm_head = nn.Linear(64, num_nmm) if num_nmm > 0 else None
        self.suffix_head = nn.Linear(64, num_suffix) if num_suffix > 0 else None
        self.rnm_head = nn.Linear(64, num_rnm) if num_rnm > 0 else None

    def forward(self, x: torch.Tensor, return_features: bool = False):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        feat = x.squeeze(-1).permute(0, 2, 1)
        gloss = self.ctc_head(feat).log_softmax(-1)
        pooled = feat.mean(dim=1)
        nmm = self.nmm_head(pooled) if self.nmm_head else None
        suffix = self.suffix_head(pooled) if self.suffix_head else None
        rnm = self.rnm_head(pooled) if self.rnm_head else None
        outputs = (gloss, nmm, suffix, rnm)
        if return_features:
            return outputs, feat
        return outputs


def build_student_model(
    name: str,
    num_classes: int,
    num_nmm: int = 0,
    num_suffix: int = 0,
    num_rnm: int = 0,
) -> nn.Module:
    if name == "stgcn":
        return STGCNStudent(3, num_classes, 544, num_nmm, num_suffix, num_rnm)
    if name == "sttn":
        return STTN(
            3,
            num_classes,
            544,
            num_layers=1,
            embed_dim=64,
            num_nmm=num_nmm,
            num_suffix=num_suffix,
            num_rnm=num_rnm,
        )
    if name == "corrnet+":
        # CorrNetPlus wraps STGCN; we reuse the student STGCN for the encoder
        return CorrNetPlus(
            3,
            num_classes,
            544,
            num_nmm=num_nmm,
            num_suffix=num_suffix,
            num_rnm=num_rnm,
        )
    if name == "mcst":
        return MCSTTransformer(
            3,
            num_classes,
            544,
            num_layers=1,
            embed_dim=64,
            num_nmm=num_nmm,
            num_suffix=num_suffix,
            num_rnm=num_rnm,
        )
    raise ValueError(f"Unknown model: {name}")


def distill(args: argparse.Namespace) -> None:
    with SignDataset(args.h5_file, args.csv_file, args.domain_labels) as ds:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inv_vocab = {v: k for k, v in ds.vocab.items()}

        teacher = build_model(
            args.model,
            len(ds.vocab),
            len(ds.nmm_vocab),
            len(ds.suffix_vocab),
            len(ds.rnm_vocab),
        )
        ckpt = torch.load(args.teacher_ckpt, map_location=device)
        teacher.load_state_dict(ckpt["model_state"])
        teacher.to(device)
        teacher.eval()

        student = build_student_model(
            args.model,
            len(ds.vocab),
            len(ds.nmm_vocab),
            len(ds.suffix_vocab),
            len(ds.rnm_vocab),
        )
        student.to(device)

        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        ce = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(student.parameters(), lr=1e-3)

        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(args.epochs):
            student.train()
            total_loss = 0.0
            for batch in dl:
                (
                    feats,
                    labels,
                    feat_lens,
                    label_lens,
                    _,
                    nmm_lbls,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = batch
                feats = feats.to(device)
                labels = labels.to(device)
                feat_lens = feat_lens.to(device)
                label_lens = label_lens.to(device)
                nmm_lbls = nmm_lbls.to(device)

                with torch.no_grad():
                    teach_logits, teach_nmm, _ = teacher(feats)

                stud_logits, stud_nmm, _ = student(feats)

                outputs = stud_logits.permute(1, 0, 2)
                targets = labels.flatten()
                loss = criterion(outputs, targets, feat_lens, label_lens)
                loss = loss + nn.functional.kl_div(
                    stud_logits.log_softmax(-1),
                    teach_logits.softmax(-1),
                    reduction="batchmean",
                )
                if stud_nmm is not None:
                    loss = loss + ce(stud_nmm, nmm_lbls)

                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()
            avg = total_loss / len(dl)
            print(f"Epoch {epoch+1}: loss {avg:.4f}")
            torch.save(
                {"model_state": student.state_dict(), "vocab": ds.vocab},
                f"checkpoints/student_epoch_{epoch+1}.pt",
            )

        teacher_wer, teacher_nmm = evaluate(teacher, dl, inv_vocab, device)
        student_wer, student_nmm = evaluate(student, dl, inv_vocab, device)
        print(f"Teacher WER {teacher_wer:.4f} | NMM {teacher_nmm:.4f}")
        print(f"Student WER {student_wer:.4f} | NMM {student_nmm:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Distil a model using KL divergence")
    p.add_argument("--h5_file", required=True)
    p.add_argument("--csv_file", required=True)
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument(
        "--model",
        default="stgcn",
        choices=["stgcn", "sttn", "corrnet+", "mcst"],
        help="Arquitectura del modelo",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--domain_labels", help="CSV con etiquetas de dominio opcional")
    args = p.parse_args()
    distill(args)
