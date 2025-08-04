from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import warnings

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


def validate_config(args: SimpleNamespace) -> None:
    """Emit warnings for out-of-range or inconsistent parameters."""

    def warn(msg: str) -> None:
        warnings.warn(msg, stacklevel=2)

    # dataset paths
    for attr in ("h5_file", "features_h5", "csv_file", "vocab", "domain_csv"):
        path = getattr(args, attr, None)
        if path and not Path(path).exists():
            warn(f"{attr} path not found: {path}")

    tckpt = getattr(args, "teacher_ckpt", None)
    if tckpt and not Path(tckpt).exists():
        warn(f"teacher_ckpt path not found: {tckpt}")

    # hyperparameters
    if getattr(args, "epochs", 1) < 1:
        warn("epochs should be >= 1")
    if not 1 <= getattr(args, "batch_size", 1) <= 1024:
        warn("batch_size should be between 1 and 1024")
    if getattr(args, "initial_length", 0) < 0:
        warn("initial_length should be >= 0")
    if not 0 <= getattr(args, "mask_prob", 0) <= 1:
        warn("mask_prob should be between 0 and 1")
    if getattr(args, "seq_len", 1) < 1:
        warn("seq_len should be >= 1")
    lr = getattr(args, "learning_rate", 0.0)
    if lr and not (1e-6 <= lr <= 1e-1):
        warn("learning_rate should be between 1e-6 and 1e-1")
    if not 1 <= getattr(args, "joint_count", 1) <= 1000:
        warn("joint_count should be between 1 and 1000")
    if getattr(args, "max_seq_len", 1) < getattr(args, "seq_len", 1):
        warn("max_seq_len should be >= seq_len")
    if getattr(args, "pose_loss_weight", 0) < 0:
        warn("pose_loss_weight should be >= 0")
    if getattr(args, "aux_loss_weight", 0) < 0:
        warn("aux_loss_weight should be >= 0")

    # adversarial training parameters
    if getattr(args, "adversarial", False):
        if not 0 <= getattr(args, "adv_mix_rate", 1.0) <= 1:
            warn("adv_mix_rate should be between 0 and 1 when adversarial is enabled")
        if getattr(args, "adv_weight", 0) < 0:
            warn("adv_weight should be >= 0 when adversarial is enabled")
        if getattr(args, "adv_steps", 1) < 1:
            warn("adv_steps should be >= 1 when adversarial is enabled")
    else:
        if (
            getattr(args, "adv_weight", 0) != 0
            or getattr(args, "adv_steps", 1) != 1
            or getattr(args, "adv_mix_rate", 1.0) != 1.0
        ):
            warn("adversarial parameters provided but adversarial is disabled")


def apply_defaults(args: SimpleNamespace, cfg: dict) -> None:
    """Fill missing attributes in ``args`` with values from ``cfg``."""
    for section in ("dataset", "checkpoints", "hyperparameters", "model"):
        defaults = cfg.get(section, {})
        for key, val in defaults.items():
            if not hasattr(args, key) or getattr(args, key) in (None, ""):
                setattr(args, key, val)
    validate_config(args)
