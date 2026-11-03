"""Top-level SensorFM: ViT-1D masked autoencoder with AIM.

Data flow:
    x         [B, T, C]
    tokens    [B, K, C, P]          patchify
    emb       [B, K, C, d]          linear + 2D positional encoding
    encoder   [B, N, d]             attention over visible tokens
    decoder   [B, N, d_dec]         mask tokens at ablated positions
    pred      [B, K, C, P]          reconstructed patches

The model returns (pred, loss_mask). The training loop computes MSE only on
loss_mask, which excludes inherited missingness.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import AimConfig, ModelConfig
from .aim import build_artificial_mask, combine_masks, ensure_visible
from .block import TransformerBlock
from .patchify import patchify, patchify_mask
from .pos_enc import SensorPositionalEncoding


class SensorFM(nn.Module):
    def __init__(self, model_cfg: ModelConfig, aim_cfg: AimConfig) -> None:
        super().__init__()
        self.cfg = model_cfg
        self.aim = aim_cfg

        self.in_proj = nn.Linear(model_cfg.patch_time, model_cfg.d_model)
        self.pe = SensorPositionalEncoding(
            model_cfg.n_channels, model_cfg.n_patches, model_cfg.d_model
        )
        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    model_cfg.d_model,
                    model_cfg.enc_heads,
                    model_cfg.mlp_ratio,
                    model_cfg.dropout,
                )
                for _ in range(model_cfg.enc_layers)
            ]
        )
        self.enc_norm = nn.LayerNorm(model_cfg.d_model)

        self.dec_in = nn.Linear(model_cfg.d_model, model_cfg.d_dec)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, model_cfg.d_dec))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder = nn.ModuleList(
            [
                TransformerBlock(
                    model_cfg.d_dec,
                    model_cfg.dec_heads,
                    model_cfg.mlp_ratio,
                    model_cfg.dropout,
                )
                for _ in range(model_cfg.dec_layers)
            ]
        )
        self.dec_norm = nn.LayerNorm(model_cfg.d_dec)
        self.out_proj = nn.Linear(model_cfg.d_dec, model_cfg.patch_time)

        # Deterministic artificial masks for v0. Each DDP rank should get its
        # own seed once distributed training is added.
        self._generator = torch.Generator(device="cpu").manual_seed(0)

    def forward(self, x: Tensor, missing: Tensor, cyclic: Tensor) -> tuple[Tensor, Tensor]:
        """
        x:       [B, T, C] float, normalized; missing positions are zero-filled
        missing: [B, T, C] bool, True = originally missing
        cyclic:  [B, K, 8] float
        returns: pred [B, K, C, P], loss_mask [B, K, C]
        """
        B, T, C = x.shape
        P = self.cfg.patch_time
        K = T // P

        tokens = patchify(x, P)  # [B, K, C, P]
        emb = self.in_proj(tokens) + self.pe(cyclic)  # [B, K, C, d]

        inherited = patchify_mask(missing, P)  # [B, K, C]
        artificial = build_artificial_mask(
            B,
            K,
            C,
            mode_probs=self.aim.mode_probs,
            random_ratio=self.aim.random_ratio,
            temporal_ratio=self.aim.temporal_ratio,
            modality_ratio=self.aim.modality_ratio,
            device=x.device,
            generator=self._generator,
        )
        full, loss_mask = combine_masks(inherited, artificial)
        full = ensure_visible(full, self._generator)

        h = emb.flatten(1, 2)  # [B, N, d]
        key_padding_mask = full.flatten(1, 2)  # [B, N]
        for block in self.encoder:
            h = block(h, key_padding_mask=key_padding_mask)
        h = self.enc_norm(h).view(B, K, C, -1)  # [B, K, C, d]

        z = self.dec_in(h)  # [B, K, C, d_dec]
        z = torch.where(full.unsqueeze(-1), self.mask_token, z)
        z = z.flatten(1, 2)  # [B, N, d_dec]
        for block in self.decoder:
            z = block(z)
        z = self.dec_norm(z)

        pred = self.out_proj(z).view(B, K, C, P)  # [B, K, C, P]
        return pred, loss_mask
