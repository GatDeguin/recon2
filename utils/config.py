from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback
    yaml = None


def load_config(path: str | Path) -> dict:
    """Load YAML configuration file with a minimal fallback parser."""
    path = Path(path)
    if not path.exists():
        return {}
    if yaml is not None:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    data: dict[str, dict] = {}
    current: dict | None = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if line.startswith(" ") and current is not None:
                key, _, val = line.strip().partition(":")
                val = val.strip()
                try:
                    current[key] = ast.literal_eval(val)
                except Exception:
                    current[key] = val
                continue
            key, _, val = line.partition(":")
            val = val.strip()
            if val:
                try:
                    data[key] = ast.literal_eval(val)
                except Exception:
                    data[key] = val
                current = None
            else:
                current = {}
                data[key] = current
    return data


def apply_defaults(args: SimpleNamespace, cfg: dict) -> None:
    """Fill missing attributes in ``args`` with values from ``cfg``."""
    for section in ("dataset", "checkpoints", "hyperparameters", "model"):
        defaults = cfg.get(section, {})
        for key, val in defaults.items():
            if not hasattr(args, key) or getattr(args, key) in (None, ""):
                setattr(args, key, val)
