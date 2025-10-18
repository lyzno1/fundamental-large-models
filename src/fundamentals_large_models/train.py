from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, ExperimentConfig, ModelConfig, OptimConfig
from .model.transformer import EncoderOnlyTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DummyLanguageModelingDataset(Dataset):
    """Simple synthetic dataset for wiring up the training loop."""

    def __init__(self, vocab_size: int, seq_len: int, length: int = 1024) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        del idx
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
        }


def collate_batch(batch: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    inputs = torch.stack([item["input_ids"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    return {"input_ids": inputs, "labels": labels, "attention_mask": mask}


def load_config(path: Path) -> Tuple[ExperimentConfig, DataConfig, ModelConfig, OptimConfig]:
    with path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    experiment = ExperimentConfig(**raw_cfg.get("experiment", {}))
    data = DataConfig(**raw_cfg.get("data", {}))
    model = ModelConfig(**raw_cfg.get("model", {}))
    optim_cfg = OptimConfig(**raw_cfg.get("optimization", {}))
    return experiment, data, model, optim_cfg


def create_dataloaders(
    data_cfg: DataConfig, optim_cfg: OptimConfig, vocab_size: int
) -> DataLoader:
    dataset = DummyLanguageModelingDataset(vocab_size, data_cfg.max_length)
    return DataLoader(
        dataset,
        batch_size=optim_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )


def train_one_epoch(
    model: EncoderOnlyTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for step, batch in enumerate(dataloader, start=1):
        inputs = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs, mask)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        steps = step
    return total_loss / max(steps, 1)


def save_metadata(output_dir: Path, experiment: ExperimentConfig, model: ModelConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"experiment": asdict(experiment), "model": asdict(model)}
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal Transformer encoder.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()

    experiment_cfg, data_cfg, model_cfg, optim_cfg = load_config(args.config)
    set_seed(experiment_cfg.seed)

    save_metadata(Path(experiment_cfg.output_dir), experiment_cfg, model_cfg)

    model = EncoderOnlyTransformer(model_cfg).to(DEVICE)

    dataloader = create_dataloaders(data_cfg, optim_cfg, model_cfg.vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=optim_cfg.learning_rate,
        betas=optim_cfg.betas,
        eps=optim_cfg.eps,
        weight_decay=optim_cfg.weight_decay,
    )

    for epoch in range(1, optim_cfg.num_epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, optim_cfg.grad_clip)
        print(f"[Epoch {epoch}] training loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
