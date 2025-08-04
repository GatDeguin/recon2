import logging
from typing import Optional

import torch

from infer import beam_search
from models.transformer_lm import load_model

logger = logging.getLogger(__name__)

# Load vocabulary if exists
vocab: dict[int, str] = {}
try:
    with open("vocab.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            vocab[i] = line.strip()
except FileNotFoundError:
    logger.warning("vocab.txt not found; vocabulary will be empty")

SOS_TOKEN = 1
EOS_TOKEN = 2

# Optional language model and beam-search params
lm_model: Optional[torch.nn.Module] = None
BEAM_SIZE = 5
LM_WEIGHT = 0.5


def init_decoder(
    lm_ckpt: Optional[str] = None, *, beam_size: int = 5, lm_weight: float = 0.5
) -> None:
    """Configure decoder with optional language model.

    Parameters
    ----------
    lm_ckpt: Optional[str]
        Path to a language-model checkpoint. If ``None`` or loading fails,
        decoding will proceed without an LM.
    beam_size: int
        Beam width for search when decoding.
    lm_weight: float
        Weight of the language model during beam search.
    """

    global lm_model, BEAM_SIZE, LM_WEIGHT
    BEAM_SIZE = beam_size
    LM_WEIGHT = lm_weight
    lm_model = None

    if lm_ckpt:
        try:
            lm_model = load_model(lm_ckpt, vocab_size=len(vocab))
        except Exception as e:  # pragma: no cover - best effort
            logger.warning("Failed to load language model from %s: %s", lm_ckpt, e)
            lm_model = None


def decode(logits: torch.Tensor) -> str:
    tokens = beam_search(logits, lm_model, BEAM_SIZE, LM_WEIGHT, SOS_TOKEN, EOS_TOKEN)
    words = [vocab.get(int(t), "<unk>") for t in tokens if int(t) not in (SOS_TOKEN, EOS_TOKEN)]
    return " ".join(words)
