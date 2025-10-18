from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Classic Transformer sinusoidal positional encoding."""

    def __init__(self, embedding_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(
        self, x: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = x.size(1)
        if self.pe.size(1) < seq_len:
            raise ValueError(
                f"Maximum supported sequence length {self.pe.size(1)} "
                f"is smaller than required length {seq_len}."
            )
        if positions is None:
            embeddings = self.pe[:, :seq_len]
        else:
            embeddings = self.pe[:, positions]
        return x + embeddings
