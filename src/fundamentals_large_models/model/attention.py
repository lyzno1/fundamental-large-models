from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        query: shape (batch, heads, seq_q, depth)
        key: shape (batch, heads, seq_k, depth)
        value: shape (batch, heads, seq_k, depth_v)
        mask: broadcastable mask, 0 for masked positions

    Returns:
        Tuple of (context, attention_weights)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    context = torch.matmul(attn, value)
    return context, attn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional bias and masking."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._reshape(q)
        k = self._reshape(k)
        v = self._reshape(v)

        if mask is not None:
            # reshape to (batch, heads=1, query_len=1, key_len) for broadcasting
            mask = mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.bool)

        context, attn = scaled_dot_product_attention(q, k, v, mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        context = self.out_proj(self.attn_dropout(context))
        return context, attn

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.size()
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
