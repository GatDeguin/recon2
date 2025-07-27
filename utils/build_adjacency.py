from __future__ import annotations
import yaml
import numpy as np
from pathlib import Path


def build_adjacency(path: str | Path) -> np.ndarray:
    """Build an adjacency matrix from a skeleton YAML config."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    offsets = {}
    offset = 0
    for name, group in data.items():
        count = int(group.get("count", 0))
        offsets[name] = offset
        group["_offset"] = offset
        offset += count

    num_nodes = offset
    A = np.eye(num_nodes, dtype=np.float32)

    for group in data.values():
        off = group.get("_offset", 0)
        for a, b in group.get("connections", []):
            ia, ib = off + int(a), off + int(b)
            A[ia, ib] = 1.0
            A[ib, ia] = 1.0
    return A
