import logging
import os
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
LM_CKPT = os.environ.get("LM_CKPT")
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "5"))
LM_WEIGHT = float(os.environ.get("LM_WEIGHT", "0.5"))
lm_model = None
if LM_CKPT:
    try:
        lm_model = load_model(LM_CKPT, vocab_size=len(vocab))
    except Exception as e:
        logger.warning("Failed to load language model from %s: %s", LM_CKPT, e)
        lm_model = None


def decode(logits: torch.Tensor) -> str:
    tokens = beam_search(logits, lm_model, BEAM_SIZE, LM_WEIGHT, SOS_TOKEN, EOS_TOKEN)
    words = [vocab.get(int(t), "<unk>") for t in tokens if int(t) not in (SOS_TOKEN, EOS_TOKEN)]
    return " ".join(words)
