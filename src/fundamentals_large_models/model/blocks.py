from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .attention import MultiHeadSelfAttention
from .feed_forward import PositionwiseFeedForward


class EncoderBlock(nn.Module):
    """Single Transformer encoder block with residual connections and LayerNorm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ff_output = self.ffn(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x, attn
