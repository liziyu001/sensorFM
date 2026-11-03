"""Turn [B, T, C] into [B, K, C, P] and back.

A token is a (patch_time, single channel) slice, never a mix of channels.
Keeping channels separate is deliberate: their units and meanings differ.
"""

from __future__ import annotations

from torch import Tensor


def patchify(x: Tensor, patch_time: int = 20) -> Tensor:
    """[B, T, C] -> [B, K, C, P]."""
    B, T, C = x.shape
    if T % patch_time != 0:
        raise ValueError(f"T={T} is not divisible by patch_time={patch_time}")
    K = T // patch_time
    return x.view(B, K, patch_time, C).permute(0, 1, 3, 2).contiguous()


def unpatchify(tokens: Tensor, T: int, C: int) -> Tensor:
    """[B, K, C, P] -> [B, T, C]."""
    B, K, C_, P = tokens.shape
    if K * P != T or C_ != C:
        raise ValueError(f"cannot unpatchify {tuple(tokens.shape)} into [B,{T},{C}]")
    return tokens.permute(0, 1, 3, 2).contiguous().view(B, T, C)


def patchify_mask(mask: Tensor, patch_time: int = 20, rule: str = "any") -> Tensor:
    """[B, T, C] bool -> [B, K, C] bool.

    rule="any":  a token is missing if any minute in the patch is missing.
    rule="all":  a token is missing only if the whole patch is missing.

    v0 uses "any": simple and conservative. It throws away partial
    observations. "all" plus minute-level attention masking is a later
    experiment.
    """
    B, T, C = mask.shape
    if T % patch_time != 0:
        raise ValueError(f"T={T} is not divisible by patch_time={patch_time}")
    K = T // patch_time
    patches = mask.view(B, K, patch_time, C)
    if rule == "any":
        return patches.any(dim=2)
    if rule == "all":
        return patches.all(dim=2)
    raise ValueError(f"unknown rule {rule!r}")
