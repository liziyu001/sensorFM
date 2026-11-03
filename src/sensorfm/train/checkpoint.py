"""Checkpoint save/load.

Store everything needed to resume exactly: model, optimizer, scheduler,
global step, and RNG states. The data loader's position is not stored in v0
(synthetic data is regenerated deterministically from the seed).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "torch_rng" in ckpt:
        torch.set_rng_state(ckpt["torch_rng"])
    if "numpy_rng" in ckpt:
        np.random.set_state(ckpt["numpy_rng"])
    if "python_rng" in ckpt:
        random.setstate(ckpt["python_rng"])
    return int(ckpt.get("step", 0))
