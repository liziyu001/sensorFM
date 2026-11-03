"""Distributed helpers (phase P2).

v0 is single-process. This module already defines the contract so that the
training script does not need to change when torchrun is introduced.

Usage:
    rank, world_size, local_rank = setup_distributed()
    ...
    cleanup_distributed()
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank). No-op without torchrun."""
    if "RANK" not in os.environ:
        return 0, 1, 0

    backend = os.environ.get("BACKEND", "nccl")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()
