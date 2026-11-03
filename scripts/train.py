"""Train SensorFM on synthetic data.

Examples:
    uv run python scripts/train.py --config configs/debug.yaml
    uv run python scripts/train.py --config configs/synthetic_xxs.yaml --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from sensorfm.config import load_config
from sensorfm.data.dataset import DayWindowDataset
from sensorfm.data.synthetic import fit_normalizer, generate_synthetic
from sensorfm.model.sensorfm import SensorFM
from sensorfm.train.loop import evaluate_loss, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None, help="override train.device")
    parser.add_argument("--steps", type=int, default=None, help="override train.max_steps")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg.train.device = args.device
    if args.steps is not None:
        cfg.train.max_steps = args.steps
    device = torch.device(cfg.train.device)

    batch = generate_synthetic(cfg.data, cfg.model.n_channels)
    mean, std = fit_normalizer(batch.x, batch.observed)
    dataset = DayWindowDataset(batch, mean, std)

    # Subject-level split: no subject appears in both train and val.
    rng = np.random.default_rng(cfg.data.seed)
    subjects = np.unique(batch.subject_id)
    n_val = max(1, int(round(args.val_fraction * subjects.size)))
    val_subjects = set(rng.choice(subjects, size=n_val, replace=False).tolist())
    is_val = np.array([s in val_subjects for s in batch.subject_id])
    train_idx = np.flatnonzero(~is_val).tolist()
    val_idx = np.flatnonzero(is_val).tolist()

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        drop_last=len(train_idx) >= cfg.train.batch_size,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    model = SensorFM(cfg.model, cfg.aim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters")
    print(f"train samples: {len(train_idx)}, val samples: {len(val_idx)}")
    print(
        f"window: {cfg.model.window_minutes} min, "
        f"patch: {cfg.model.patch_time} min, "
        f"tokens: {cfg.model.n_tokens}"
    )

    train(cfg.train, model, train_loader, device)

    if val_idx:
        val_loss = evaluate_loss(model, val_loader, device)
        print(f"val loss: {val_loss:.5f}")

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"saved {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
