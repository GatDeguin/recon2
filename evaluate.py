#!/usr/bin/env python
import argparse
import json
from functools import partial
from collections import defaultdict
from typing import Dict, List

import torch
from torch.utils.data import DataLoader


def _levenshtein(a: List[str], b: List[str]) -> int:
    """Compute Levenshtein distance between two sequences."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def compute_metrics(model, dl, inv_vocab, device):
    """Compute WER, CER and per-class accuracies for a dataset."""

    model.eval()
    total_words = 0
    word_errs = 0
    total_chars = 0
    char_errs = 0
    class_correct: Dict[str, int] = defaultdict(int)
    class_total: Dict[str, int] = defaultdict(int)
    skip_tokens = {0, 1, 2}

    with torch.no_grad():
        for batch in dl:
            (
                feats,
                labels,
                feat_lens,
                label_lens,
                domains,
                nmm_lbls,
                suf_lbls,
                rnm_lbls,
                per_lbls,
                num_lbls,
                tense_lbls,
                aspect_lbls,
                mode_lbls,
            ) = batch
            feats = feats.to(device)
            labels = labels.to(device)
            nmm_lbls = nmm_lbls.to(device)
            suf_lbls = suf_lbls.to(device)
            rnm_lbls = rnm_lbls.to(device)
            per_lbls = per_lbls.to(device)
            num_lbls = num_lbls.to(device)
            tense_lbls = tense_lbls.to(device)
            aspect_lbls = aspect_lbls.to(device)
            mode_lbls = mode_lbls.to(device)

            outputs = model(feats)
            (
                gloss_logits,
                nmm_logits,
                suf_logits,
                rnm_logits,
                per_logits,
                num_logits,
                tense_logits,
                aspect_logits,
                mode_logits,
            ) = outputs

            preds = gloss_logits.argmax(-1)
            for p, t in zip(preds, labels):
                pred_tokens: List[str] = []
                last = 0
                for tok in p.tolist():
                    if tok not in skip_tokens and tok != last:
                        pred_tokens.append(inv_vocab.get(tok, ""))
                    last = tok
                tgt_tokens = [inv_vocab.get(int(x), "") for x in t if int(x) not in skip_tokens]
                hyp = " ".join(pred_tokens).strip()
                ref = " ".join(tgt_tokens).strip()

                ref_words = ref.split()
                hyp_words = hyp.split()
                word_errs += _levenshtein(ref_words, hyp_words)
                total_words += len(ref_words)

                ref_chars = list(ref.replace(" ", ""))
                hyp_chars = list(hyp.replace(" ", ""))
                char_errs += _levenshtein(ref_chars, hyp_chars)
                total_chars += len(ref_chars)

            if nmm_logits is not None:
                pred = nmm_logits.argmax(-1)
                class_correct["nmm"] += (pred.cpu() == nmm_lbls.cpu()).sum().item()
                class_total["nmm"] += len(nmm_lbls)
            if suf_logits is not None:
                pred = suf_logits.argmax(-1)
                class_correct["suffix"] += (pred.cpu() == suf_lbls.cpu()).sum().item()
                class_total["suffix"] += len(suf_lbls)
            if rnm_logits is not None:
                pred = rnm_logits.argmax(-1)
                class_correct["rnm"] += (pred.cpu() == rnm_lbls.cpu()).sum().item()
                class_total["rnm"] += len(rnm_lbls)
            if per_logits is not None:
                pred = per_logits.argmax(-1)
                class_correct["person"] += (pred.cpu() == per_lbls.cpu()).sum().item()
                class_total["person"] += len(per_lbls)
            if num_logits is not None:
                pred = num_logits.argmax(-1)
                class_correct["number"] += (pred.cpu() == num_lbls.cpu()).sum().item()
                class_total["number"] += len(num_lbls)
            if tense_logits is not None:
                pred = tense_logits.argmax(-1)
                class_correct["tense"] += (pred.cpu() == tense_lbls.cpu()).sum().item()
                class_total["tense"] += len(tense_lbls)
            if aspect_logits is not None:
                pred = aspect_logits.argmax(-1)
                class_correct["aspect"] += (pred.cpu() == aspect_lbls.cpu()).sum().item()
                class_total["aspect"] += len(aspect_lbls)
            if mode_logits is not None:
                pred = mode_logits.argmax(-1)
                class_correct["mode"] += (pred.cpu() == mode_lbls.cpu()).sum().item()
                class_total["mode"] += len(mode_lbls)

    wer = word_errs / total_words if total_words else 0.0
    cer = char_errs / total_chars if total_chars else 0.0
    class_acc = {k: class_correct[k] / class_total[k] if class_total[k] else 0.0 for k in class_correct}
    return wer, cer, class_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--h5_file", required=True, help="HDF5 file with landmarks")
    parser.add_argument("--csv_file", required=True, help="CSV file with labels")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--model", choices=["stgcn", "sttn", "corrnet+", "mcst"], default="stgcn")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--segments", action="store_true", help="Use segmented samples")
    parser.add_argument("--include_openface", action="store_true", help="Include extra OpenFace features")
    args = parser.parse_args()

    from train import SignDataset, collate, build_model

    with SignDataset(
        args.h5_file,
        args.csv_file,
        segments=args.segments,
        include_openface=args.include_openface,
    ) as ds:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate))
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
        wer, cer, class_acc = compute_metrics(model, dl, inv_vocab, device)
        print(json.dumps({"wer": wer, "cer": cer, "class_acc": class_acc}, indent=2))


if __name__ == "__main__":
    main()
