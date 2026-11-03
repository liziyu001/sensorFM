"""Imputation and forecasting evaluation.

Implement in phase P0/P1, after the training loop can overfit one batch.

Tasks to support (paper M.3.5):
    - random imputation: mask 80% of tokens
    - temporal interpolation: contiguous 10/30/60 min in the middle
    - temporal extrapolation: contiguous 10/30/60 min at the end
    - signal imputation: drop 2/6/12/26 whole channels

For each task, compare SensorFM against mean fill, nearest-neighbor fill, and
linear interpolation on the same masked positions.
"""

from __future__ import annotations

from torch import nn


def evaluate_imputation(model: nn.Module, loader, device) -> dict[str, float]:
    raise NotImplementedError("Phase P0/P1: implement mask-and-reconstruct evaluation.")
