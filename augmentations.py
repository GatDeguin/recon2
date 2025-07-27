"""Data augmentation utilities for sign sequences."""

from typing import Callable

import torch
from torch.nn import functional as F


def temporal_mixup(seq_a: torch.Tensor, seq_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Mix two sequences along the temporal dimension.

    Parameters
    ----------
    seq_a, seq_b : torch.Tensor
        Input sequences with shape ``(C, T, V)``.
    alpha : float
        Beta distribution parameter.

    Returns
    -------
    torch.Tensor
        Mixed sequence with the same shape as the inputs.
    """
    beta = torch.distributions.Beta(alpha, alpha)
    lam = beta.sample().to(seq_a.device)
    return lam * seq_a + (1.0 - lam) * seq_b


def speed_perturbation(seq: torch.Tensor, rate: float) -> torch.Tensor:
    """Change sequence speed using linear interpolation.

    Parameters
    ----------
    seq : torch.Tensor
        Sequence with shape ``(C, T, V)``.
    rate : float
        Speed factor. Values > 1 speed up, < 1 slow down.

    Returns
    -------
    torch.Tensor
        Sequence with length ``round(T * rate)``.
    """
    c, t, v = seq.shape
    inp = seq.permute(2, 0, 1).reshape(1, c * v, t)
    out = F.interpolate(inp, scale_factor=rate, mode="linear", align_corners=False)
    new_t = out.shape[-1]
    out = out.reshape(v, c, new_t).permute(1, 2, 0)
    return out


def overlay_gan_background(seq: torch.Tensor, gan_model: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Overlay GAN-generated background over a sequence of frames.

    The ``gan_model`` callable must return a tensor with the same shape as ``seq``.
    Both tensors are added and clipped to the valid range ``[0, 1]``.
    """
    bg = gan_model(seq)
    return (seq + bg).clamp(0.0, 1.0)

