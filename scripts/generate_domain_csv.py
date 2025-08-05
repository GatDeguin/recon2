#!/usr/bin/env python
"""Generate domain label CSV for adversarial training.

This utility combines the ``meta.csv`` files from multiple datasets and
assigns a numeric domain id to each sample. The resulting CSV contains two
columns separated by ``;``: ``id`` and ``domain``. Domains are enumerated in
the same order as the input paths starting from 0.

Example
-------
    python scripts/generate_domain_csv.py data/domains.csv data/lsa_t/train data/phoenix/train
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def _read_ids(path: str | Path) -> Iterable[str]:
    """Return the sequence ids from a ``meta.csv`` located at *path*.

    ``path`` can be either a directory containing ``meta.csv`` or the path to
    the CSV file itself. The file must contain an ``id`` column.
    """

    p = Path(path)
    if p.is_dir():
        p = p / "meta.csv"
    if not p.is_file():
        raise FileNotFoundError(f"meta.csv not found in {path}")
    df = pd.read_csv(p, sep=";")
    if "id" not in df.columns:
        raise ValueError(f"'id' column missing in {p}")
    return df["id"].astype(str)


def generate(output_csv: str, inputs: list[str]) -> None:
    rows = []
    for domain, inp in enumerate(inputs):
        ids = _read_ids(inp)
        rows.append(pd.DataFrame({"id": ids, "domain": domain}))
    out_df = pd.concat(rows, ignore_index=True)
    out_df.to_csv(output_csv, sep=";", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Build domain CSV for DANN")
    p.add_argument("output_csv", help="Path to output CSV")
    p.add_argument(
        "inputs",
        nargs="+",
        help="Dataset directories or meta.csv files. Order defines domain id",
    )
    args = p.parse_args()
    generate(args.output_csv, args.inputs)


if __name__ == "__main__":  # pragma: no cover - CLI utility
    main()
