"""Generate a synthetic dataset and save it as .npz.

Example:
    uv run python scripts/make_synthetic.py --config configs/debug.yaml \
        --out data/debug.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sensorfm.config import load_config
from sensorfm.data.synthetic import fit_normalizer, generate_synthetic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="data/synthetic.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    batch = generate_synthetic(cfg.data, cfg.model.n_channels)
    mean, std = fit_normalizer(batch.x, batch.observed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x=batch.x,
        observed=batch.observed,
        cyclic=batch.cyclic,
        subject_id=batch.subject_id,
        mean=mean,
        std=std,
        **{f"labels_{k}": v for k, v in batch.labels.items()},
    )

    missing_rate = 1.0 - batch.observed.mean()
    print(f"wrote {out}")
    print(f"  samples: {len(batch)}  shape: {batch.x.shape}")
    print(f"  subjects: {batch.subject_id.max() + 1}")
    print(f"  missing rate: {missing_rate:.3f}")


if __name__ == "__main__":
    main()
