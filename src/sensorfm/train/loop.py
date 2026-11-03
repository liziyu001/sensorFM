"""Minimal single-process training loop.

Deliberately thin: it knows about batches, optimizer steps, and logging, but
not about data layout or mask construction. DDP and gradient accumulation are
later phases; keep this version readable.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import Tensor

from ..config import TrainConfig
from ..model.patchify import patchify


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mse(pred: Tensor, target: Tensor, loss_mask: Tensor) -> Tensor:
    """MSE only where loss_mask is True.

    pred, target: [B, K, C, P]
    loss_mask:    [B, K, C]
    """
    m = loss_mask.unsqueeze(-1).expand_as(pred)
    se = (pred - target).pow(2)
    return (se * m).sum() / m.sum().clamp_min(1)


def train(
    cfg: TrainConfig,
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> torch.nn.Module:
    set_seed(cfg.seed)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=cfg.max_steps,
        pct_start=cfg.warmup_pct,
    )

    try:
        n_batches = len(loader)
    except TypeError:
        n_batches = None
    if n_batches == 0:
        raise ValueError("loader is empty; increase data or lower batch_size")

    use_amp = cfg.amp and device.type == "cuda"
    step = 0
    while step < cfg.max_steps:
        for batch in loader:
            x, missing, cyclic = (t.to(device) for t in batch)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                pred, loss_mask = model(x, missing, cyclic)
                target = patchify(x, model.cfg.patch_time)
                loss = masked_mse(pred, target, loss_mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            if step % cfg.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"step {step:6d} | loss {loss.item():.5f} "
                    f"| mask {loss_mask.float().mean().item():.3f} "
                    f"| lr {lr:.2e}"
                )

            step += 1
            if step >= cfg.max_steps:
                break

    return model


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader, device: torch.device) -> float:
    """Average masked MSE over a loader.

    Note: AIM samples fresh artificial masks on every call, so this is a
    stochastic estimate. For a deterministic validation curve, add a fixed
    mask mode later.
    """
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        x, missing, cyclic = (t.to(device) for t in batch)
        pred, loss_mask = model(x, missing, cyclic)
        target = patchify(x, model.cfg.patch_time)
        total += masked_mse(pred, target, loss_mask).item()
        count += 1
    model.train()
    return total / max(1, count)
