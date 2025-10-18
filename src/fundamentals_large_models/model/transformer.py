from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from ..config import ModelConfig
from .blocks import EncoderBlock
from .positional_encoding import SinusoidalPositionalEncoding


class EncoderOnlyTransformer(nn.Module):
    """Minimal encoder-only Transformer suitable for language modeling."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(
            config.embedding_dim, config.max_position_embeddings
        )
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    config.embedding_dim,
                    config.num_heads,
                    config.ff_dim,
                    config.dropout,
                    config.layer_norm_eps,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, attention_mask)
            attn_maps.append(attn)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, torch.stack(attn_maps)
