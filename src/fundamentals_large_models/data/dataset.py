from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from datasets import load_dataset  # type: ignore


@dataclass
class TokenizerOutput:
    input_ids: list[int]
    attention_mask: list[int]


def build_tokenizer(lowercase: bool = True) -> Callable[[str], TokenizerOutput]:
    """Create a simple whitespace tokenizer for quick experiments."""

    def tokenize(text: str) -> TokenizerOutput:
        if lowercase:
            text = text.lower()
        tokens = text.split()
        return TokenizerOutput(
            input_ids=[hash(token) % 50000 for token in tokens],
            attention_mask=[1] * len(tokens),
        )

    return tokenize


def load_small_corpus(
    dataset_name: str,
    subset: Optional[str],
    text_field: str,
    split: str,
) -> Dict[str, list[str]]:
    """Download a small text corpus via Hugging Face datasets."""
    dataset = load_dataset(dataset_name, subset, split=split)
    return dataset[text_field]
