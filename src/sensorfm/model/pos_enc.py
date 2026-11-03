"""2D additive positional encoding: (time patch index, channel index).

The time half is sinusoidal over patch index, except for 8 dimensions that
carry cyclic calendar features (minute, hour, weekday, yearday). The channel
half is a learned embedding: channel identity is categorical, not positional.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sinusoidal_encoding(n: int, d: int) -> Tensor:
    """[n, d] standard sinusoidal table. Requires d even."""
    if d % 2 != 0:
        raise ValueError("sinusoidal encoding width must be even")
    position = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SensorPositionalEncoding(nn.Module):
    """cyclic: [B, K, 8] -> PE: [B, K, C, d_model]."""

    def __init__(self, n_channels: int, n_patches: int, d_model: int) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")
        time_dim = d_model // 2 - 8
        if time_dim <= 0:
            raise ValueError("d_model is too small for 8 cyclic features")
        if time_dim % 2 != 0:
            raise ValueError("time_dim must be even; choose another d_model")
        self.n_channels = n_channels
        self.n_patches = n_patches
        self.channel = nn.Embedding(n_channels, d_model // 2)
        self.register_buffer("time_sin", sinusoidal_encoding(n_patches, time_dim), persistent=False)

    def forward(self, cyclic: Tensor) -> Tensor:
        B, K, _ = cyclic.shape
        if K != self.n_patches:
            raise ValueError(f"expected K={self.n_patches}, got {K}")
        C = self.n_channels

        # [B, C, d/2] -> [B, K, C, d/2]
        channel = self.channel.weight.unsqueeze(0).expand(B, -1, -1)
        channel = channel.unsqueeze(1).expand(-1, K, -1, -1)

        # [K, d/2-8] -> [B, K, C, d/2-8]
        time = self.time_sin[:K].unsqueeze(0).unsqueeze(2).expand(B, -1, C, -1)

        # [B, K, 8] -> [B, K, C, 8]
        cal = cyclic.unsqueeze(2).expand(-1, -1, C, -1)

        time = torch.cat([time, cal], dim=-1)  # [B, K, C, d/2]
        return torch.cat([channel, time], dim=-1)  # [B, K, C, d]
