"""Typed configuration objects loaded from YAML.

Design note: configuration is a plain dataclass tree, not a framework.
Unknown YAML keys are ignored on purpose so that configs can carry comments
and future fields without breaking older code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Architecture hyperparameters.

    Shapes (batch B, window T, channels C, patch P):
        T = window_minutes, K = T / patch_time, N = K * C
        tokens: [B, K, C, P] -> [B, N, d_model]
    """

    n_channels: int = 34
    window_minutes: int = 1440
    patch_time: int = 20
    d_model: int = 64
    enc_layers: int = 2
    enc_heads: int = 1
    mlp_ratio: float = 4.0
    d_dec: int = 48
    dec_layers: int = 1
    dec_heads: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.window_minutes % self.patch_time != 0:
            raise ValueError("window_minutes must be divisible by patch_time")
        if self.d_model % 2 != 0:
            raise ValueError("d_model must be even (half for channels, half for time)")
        if self.d_model // 2 - 8 <= 0:
            raise ValueError("d_model must be large enough to hold 8 cyclic features")

    @property
    def n_patches(self) -> int:
        return self.window_minutes // self.patch_time

    @property
    def n_tokens(self) -> int:
        return self.n_patches * self.n_channels


@dataclass
class AimConfig:
    """Artificial-mask ratios for Adaptive and Inherited Masking.

    The three modes simulate: random dropouts, device removal (temporal
    blocks), and missing sensors (modality blocks).
    """

    random_ratio: float = 0.8
    temporal_ratio: float = 0.5
    modality_ratio: float = 0.5
    mode_probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)


@dataclass
class DataConfig:
    """Synthetic data generator settings (real-data loaders live in data/)."""

    n_subjects: int = 64
    days_per_subject: int = 4
    window_minutes: int = 1440
    patch_time: int = 20
    seed: int = 0
    missing_mcar_rate: float = 0.02
    missing_block_prob: float = 0.3
    missing_block_hours: tuple[float, float] = (1.0, 4.0)
    missing_modality_prob: float = 0.2


@dataclass
class TrainConfig:
    """Training loop settings. v0 is single-process; DDP is a later phase."""

    batch_size: int = 8
    lr: float = 5e-4
    weight_decay: float = 1e-4
    max_steps: int = 200
    warmup_pct: float = 0.05
    grad_clip: float = 1.0
    amp: bool = False
    log_every: int = 10
    num_workers: int = 0
    device: str = "cpu"
    out_dir: str = "runs/debug"
    seed: int = 0


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    aim: AimConfig = field(default_factory=AimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if self.data.window_minutes != self.model.window_minutes:
            raise ValueError("data.window_minutes must match model.window_minutes")
        if self.data.patch_time != self.model.patch_time:
            raise ValueError("data.patch_time must match model.patch_time")


_TUPLE_FIELDS = {"mode_probs", "missing_block_hours"}


def _build(cls: type, raw: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        if f.name in _TUPLE_FIELDS and value is not None:
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config into the typed dataclass tree."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ExperimentConfig(
        model=_build(ModelConfig, raw.get("model", {})),
        data=_build(DataConfig, raw.get("data", {})),
        aim=_build(AimConfig, raw.get("aim", {})),
        train=_build(TrainConfig, raw.get("train", {})),
    )
