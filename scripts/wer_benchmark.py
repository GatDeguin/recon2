#!/usr/bin/env python
"""Compute WER and CER for multiple model predictions."""
import argparse
import csv
import json
from typing import Dict, List


def _levenshtein(a: List[str], b: List[str]) -> int:
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


def _rates(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    word_errs = char_errs = total_words = total_chars = 0
    for ref, hyp in zip(refs, hyps):
        r_words, h_words = ref.split(), hyp.split()
        word_errs += _levenshtein(r_words, h_words)
        total_words += len(r_words)
        r_chars = list(ref.replace(" ", ""))
        h_chars = list(hyp.replace(" ", ""))
        char_errs += _levenshtein(r_chars, h_chars)
        total_chars += len(r_chars)
    wer = word_errs / total_words if total_words else 0.0
    cer = char_errs / total_chars if total_chars else 0.0
    return {"wer": wer, "cer": cer}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark models with WER/CER")
    parser.add_argument("csv", help="CSV file with reference and prediction columns")
    parser.add_argument("--models", nargs="+", required=True, help="Model column names")
    parser.add_argument("--output", help="Optional JSON output file")
    args = parser.parse_args()

    refs: List[str] = []
    preds: Dict[str, List[str]] = {m: [] for m in args.models}
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            refs.append(row["reference"])
            for m in args.models:
                preds[m].append(row[m])

    results = {m: _rates(refs, hyps) for m, hyps in preds.items()}
    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
