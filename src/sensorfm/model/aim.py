"""Adaptive and Inherited Masking (AIM).

Core idea:
    full      = inherited | artificial     # what the model sees as missing
    loss_mask = artificial & ~inherited    # what the model is asked to predict

We never compute loss on inherited missingness because its ground truth does
not exist. That single choice is what makes missingness a signal instead of
noise.
"""

from __future__ import annotations

import torch
from torch import Tensor


def build_artificial_mask(
    B: int,
    K: int,
    C: int,
    *,
    mode_probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    random_ratio: float = 0.8,
    temporal_ratio: float = 0.5,
    modality_ratio: float = 0.5,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return [B, K, C] bool, True = artificially masked.

    Each sample picks one of three modes:
      0: random token dropout
      1: contiguous temporal block (device removed)
      2: whole-channel block (sensor not present)
    """
    if len(mode_probs) != 3:
        raise ValueError("mode_probs must have three entries")
    art = torch.zeros(B, K, C, dtype=torch.bool)
    probs = torch.tensor(mode_probs, dtype=torch.float32)
    modes = torch.multinomial(probs, B, replacement=True, generator=generator)

    for b in range(B):
        mode = int(modes[b].item())
        if mode == 0:
            art[b] = torch.rand(K, C, generator=generator) < random_ratio
        elif mode == 1:
            length = max(1, int(round(K * temporal_ratio)))
            start = int(torch.randint(0, K - length + 1, (1,), generator=generator))
            art[b, start : start + length, :] = True
        else:
            n_channels = max(1, int(round(C * modality_ratio)))
            channels = torch.randperm(C, generator=generator)[:n_channels]
            art[b, :, channels] = True

    if device is not None:
        art = art.to(device)
    return art


def combine_masks(inherited: Tensor, artificial: Tensor) -> tuple[Tensor, Tensor]:
    """Return (full_mask, loss_mask); both [B, K, C] bool."""
    full = inherited | artificial
    loss_mask = artificial & ~inherited
    return full, loss_mask


def ensure_visible(full: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Guarantee at least one visible token per sample.

    An all-masked row would make softmax(attention) undefined (all -inf).
    We un-mask one random token; if its value was missing it is zero-filled,
    which is acceptable for a guard.
    """
    B, K, C = full.shape
    all_missing = full.flatten(1).all(dim=1)
    if not bool(all_missing.any()):
        return full

    full = full.clone()
    for b in torch.nonzero(all_missing, as_tuple=False).flatten().tolist():
        k = int(torch.randint(0, K, (1,), generator=generator))
        c = int(torch.randint(0, C, (1,), generator=generator))
        full[b, k, c] = False
    return full
