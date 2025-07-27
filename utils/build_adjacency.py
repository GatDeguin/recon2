from __future__ import annotations
import numpy as np
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - fallback if PyYAML is missing
    yaml = None


def _load_config(path: Path) -> dict:
    """Load YAML configuration or use a minimal fallback parser."""
    if yaml is not None:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    data: dict[str, dict] = {}
    current: str | None = None
    anchors: dict[str, list] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if not line.startswith(" "):
                key = line.split(":", 1)[0]
                current = key
                data[current] = {}
                continue
            if current is None:
                continue
            item = line.strip()
            if item.startswith("count:"):
                data[current]["count"] = int(item.split(":", 1)[1].strip())
            elif item.startswith("connections:"):
                if "&" in item:
                    name = item.split("&", 1)[1].strip()
                    anchors[name] = []
                    data[current]["connections"] = anchors[name]
                elif "*" in item:
                    alias = item.split("*", 1)[1].strip()
                    data[current]["connections"] = list(anchors.get(alias, []))
                else:
                    data[current]["connections"] = []
            elif item.startswith("-"):
                parts = item.strip("-[] ").split(",")
                if len(parts) >= 2:
                    a, b = int(parts[0]), int(parts[1])
                    data[current].setdefault("connections", []).append([a, b])
    return data


def build_adjacency(path: str | Path) -> np.ndarray:
    """Build an adjacency matrix from a skeleton YAML config."""
    path = Path(path)
    data = _load_config(path)

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
