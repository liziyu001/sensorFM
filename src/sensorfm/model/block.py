"""Pre-LN Transformer block.

We hand-roll this instead of using nn.TransformerEncoder because we need
explicit per-token padding masks and later a custom loss mask. The block is
small enough that the extra clarity is worth it.
"""

from __future__ import annotations

from torch import Tensor, nn


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """x: [B, N, d]; key_padding_mask: [B, N] bool, True = ignore."""
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x
