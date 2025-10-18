from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None

from ..config import DataConfig, TokenizerConfig


class BasicTokenizer:
    """Whitespace tokenizer with frequency-based vocabulary."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self.lowercase = config.lowercase
        self.min_freq = max(config.min_freq, 1)
        self.pad_token = config.pad_token
        self.unk_token = config.unk_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token

        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return text.strip().split()

    def build_vocab(self, texts: Iterable[str]) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        specials = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.itos = []
        seen = set()
        for token in specials:
            if token not in seen:
                self.itos.append(token)
                seen.add(token)

        for token, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq < self.min_freq:
                continue
            if token in seen:
                continue
            self.itos.append(token)
            seen.add(token)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self._tokenize(text)
        if add_special_tokens:
            tokens = [self.bos_token, *tokens, self.eos_token]
        unk_id = self.stoi.get(self.unk_token, 1)
        return [self.stoi.get(token, unk_id) for token in tokens]

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def save(self, path: Path) -> None:
        payload = {
            "config": {
                "lowercase": self.lowercase,
                "min_freq": self.min_freq,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
            },
            "itos": self.itos,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BasicTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = TokenizerConfig(**data["config"])
        tokenizer = cls(cfg)
        tokenizer.itos = list(data["itos"])
        tokenizer.stoi = {token: idx for idx, token in enumerate(tokenizer.itos)}
        return tokenizer


def build_tokenizer(config: TokenizerConfig, training_texts: Iterable[str]) -> BasicTokenizer:
    """Create or load a tokenizer based on training data."""

    if config.vocab_file:
        vocab_path = Path(config.vocab_file)
        if vocab_path.exists():
            return BasicTokenizer.load(vocab_path)

    tokenizer = BasicTokenizer(config)
    tokenizer.build_vocab(training_texts)

    if config.vocab_file:
        tokenizer.save(Path(config.vocab_file))

    return tokenizer


class LanguageModelingDataset(Dataset):
    """Chunked language modeling dataset that returns fixed-length sequences."""

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: BasicTokenizer,
        seq_len: int,
        limit_examples: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_id = tokenizer.pad_id
        self.examples = self._build_examples(list(texts), limit_examples)
        if not self.examples:
            raise ValueError("Dataset construction produced no examples. Check input texts.")

    def _build_examples(
        self, texts: List[str], limit_examples: Optional[int]
    ) -> List[List[int]]:
        examples: List[List[int]] = []
        max_tokens = self.seq_len + 1
        for text in texts:
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) < 2:
                continue
            start = 0
            while start + 1 < len(token_ids):
                chunk = token_ids[start : start + max_tokens]
                if len(chunk) < max_tokens:
                    chunk = chunk + [self.pad_id] * (max_tokens - len(chunk))
                examples.append(chunk)
                if limit_examples is not None and len(examples) >= limit_examples:
                    return examples
                start += self.seq_len
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.examples[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def _load_local_split(split_path: Path) -> List[str]:
    if not split_path.exists():
        raise FileNotFoundError(f"Expected split file at {split_path}, but it does not exist.")
    lines = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def load_text_splits(config: DataConfig) -> Dict[str, List[str]]:
    """Load dataset splits either from local files or Hugging Face datasets."""

    if config.dataset_name == "local_text":
        if config.data_dir is None:
            raise ValueError("data_dir must be specified for local_text dataset.")
        data_dir = Path(config.data_dir)
        splits = {
            "train": _load_local_split(data_dir / f"{config.train_split}.txt"),
            "validation": _load_local_split(data_dir / f"{config.val_split}.txt"),
        }
        if config.test_split:
            test_path = data_dir / f"{config.test_split}.txt"
            if test_path.exists():
                splits["test"] = _load_local_split(test_path)
        return splits

    if load_dataset is None:
        raise RuntimeError(
            "datasets package is unavailable; install `datasets` or use the local_text dataset."
        )

    splits: Dict[str, List[str]] = {}
    split_mappings = [
        (config.train_split, "train"),
        (config.val_split, "validation"),
        (config.test_split, "test"),
    ]
    for split_name, target_name in split_mappings:
        if split_name is None:
            continue
        dataset = load_dataset(config.dataset_name, config.subset, split=split_name)
        splits[target_name] = dataset[config.text_field]
    return splits
