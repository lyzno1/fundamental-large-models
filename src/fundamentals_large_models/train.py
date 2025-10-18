from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
    TokenizerConfig,
)
from .data.dataset import BasicTokenizer, LanguageModelingDataset, build_tokenizer, load_text_splits
from .model.transformer import EncoderOnlyTransformer

plt.switch_backend("agg")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(
    path: Path,
) -> Tuple[ExperimentConfig, DataConfig, ModelConfig, OptimConfig, TokenizerConfig]:
    with path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    experiment = ExperimentConfig(**raw_cfg.get("experiment", {}))
    data = DataConfig(**raw_cfg.get("data", {}))
    model = ModelConfig(**raw_cfg.get("model", {}))
    optim_cfg = OptimConfig(**raw_cfg.get("optimization", {}))
    tokenizer_cfg = TokenizerConfig(**raw_cfg.get("tokenizer", {}))

    experiment.output_dir = Path(experiment.output_dir)
    if data.data_dir is not None:
        data.data_dir = Path(data.data_dir)
    if tokenizer_cfg.vocab_file is not None:
        tokenizer_cfg.vocab_file = Path(tokenizer_cfg.vocab_file)

    return experiment, data, model, optim_cfg, tokenizer_cfg


def create_scheduler(
    optimizer: optim.Optimizer, optim_cfg: OptimConfig, total_steps: int
) -> Optional[LambdaLR]:
    if total_steps <= 0:
        return None

    warmup_steps = max(optim_cfg.warmup_steps, 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress_steps = total_steps - warmup_steps
        if progress_steps <= 0:
            return 1.0
        progress = (step - warmup_steps) / progress_steps
        progress = min(max(progress, 0.0), 1.0)
        if optim_cfg.scheduler == "linear":
            return max(0.0, 1.0 - progress)
        if optim_cfg.scheduler == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def build_datasets_and_tokenizer(
    data_cfg: DataConfig,
    tokenizer_cfg: TokenizerConfig,
) -> Tuple[LanguageModelingDataset, Optional[LanguageModelingDataset], BasicTokenizer]:
    splits = load_text_splits(data_cfg)
    train_texts = splits["train"]
    tokenizer = build_tokenizer(tokenizer_cfg, train_texts)

    train_dataset = LanguageModelingDataset(
        train_texts,
        tokenizer,
        data_cfg.max_length,
        limit_examples=data_cfg.limit_examples,
    )

    val_dataset: Optional[LanguageModelingDataset] = None
    if "validation" in splits and splits["validation"]:
        val_limit = None
        if data_cfg.limit_examples is not None:
            val_limit = max(1, data_cfg.limit_examples // 4)
        try:
            val_dataset = LanguageModelingDataset(
                splits["validation"],
                tokenizer,
                data_cfg.max_length,
                limit_examples=val_limit,
            )
        except ValueError:
            val_dataset = None

    return train_dataset, val_dataset, tokenizer


def create_dataloaders(
    train_dataset: LanguageModelingDataset,
    val_dataset: Optional[LanguageModelingDataset],
    optim_cfg: OptimConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=optim_cfg.batch_size,
        shuffle=True,
    )
    val_loader: Optional[DataLoader] = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=optim_cfg.batch_size,
            shuffle=False,
        )
    return train_loader, val_loader


def move_batch_to_device(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.to(DEVICE) for key, value in batch.items()}


def train_one_epoch(
    model: EncoderOnlyTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[LambdaLR],
    grad_clip: float,
    log_interval: int,
    start_step: int,
) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    steps = 0
    global_step = start_step

    for step, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        steps = step
        global_step += 1

        if log_interval > 0 and step % log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  step {step:04d} | loss {loss.item():.4f} | lr {current_lr:.2e}",
                flush=True,
            )

    avg_loss = total_loss / max(steps, 1)
    return avg_loss, global_step


@torch.no_grad()
def evaluate(
    model: EncoderOnlyTransformer,
    dataloader: Optional[DataLoader],
    criterion: nn.Module,
) -> Optional[float]:
    if dataloader is None:
        return None

    model.eval()
    total_loss = 0.0
    steps = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch)
        logits, _ = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
        )
        total_loss += loss.item()
        steps += 1

    if steps == 0:
        return None
    return total_loss / steps


def save_metadata(
    output_dir: Path,
    experiment: ExperimentConfig,
    data: DataConfig,
    model: ModelConfig,
    optim_cfg: OptimConfig,
    tokenizer_cfg: TokenizerConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": asdict(experiment),
        "data": asdict(data),
        "model": asdict(model),
        "optimization": asdict(optim_cfg),
        "tokenizer": asdict(tokenizer_cfg),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)


def save_training_artifacts(
    output_dir: Path,
    train_losses: List[float],
    val_losses: List[Optional[float]],
    model: EncoderOnlyTransformer,
    optimizer: optim.Optimizer,
    scheduler: Optional[LambdaLR],
    tokenizer: BasicTokenizer,
    tokenizer_path: Path,
) -> None:
    metrics = {"train_loss": train_losses, "val_loss": val_losses}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_losses, label="train")
    if any(v is not None for v in val_losses):
        ax.plot(
            epochs,
            [v if v is not None else float("nan") for v in val_losses],
            label="validation",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png")
    plt.close(fig)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        output_dir / "model.pt",
    )

    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Transformer encoder on a small corpus.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()

    (
        experiment_cfg,
        data_cfg,
        model_cfg,
        optim_cfg,
        tokenizer_cfg,
    ) = load_config(args.config)

    set_seed(experiment_cfg.seed)

    output_dir = Path(experiment_cfg.output_dir)
    if tokenizer_cfg.vocab_file is None:
        tokenizer_cfg.vocab_file = output_dir / "tokenizer.json"

    train_dataset, val_dataset, tokenizer = build_datasets_and_tokenizer(data_cfg, tokenizer_cfg)

    model_cfg.vocab_size = tokenizer.vocab_size

    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, optim_cfg)

    model = EncoderOnlyTransformer(model_cfg).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=optim_cfg.learning_rate,
        betas=optim_cfg.betas,
        eps=optim_cfg.eps,
        weight_decay=optim_cfg.weight_decay,
    )

    total_steps = len(train_loader) * optim_cfg.num_epochs
    scheduler = create_scheduler(optimizer, optim_cfg, total_steps)

    tokenizer_save_path = Path(tokenizer_cfg.vocab_file)

    save_metadata(output_dir, experiment_cfg, data_cfg, model_cfg, optim_cfg, tokenizer_cfg)

    train_losses: List[float] = []
    val_losses: List[Optional[float]] = []
    global_step = 0

    for epoch in range(1, optim_cfg.num_epochs + 1):
        print(f"Epoch {epoch}/{optim_cfg.num_epochs}")
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            optim_cfg.grad_clip,
            optim_cfg.log_interval,
            global_step,
        )
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss)

        status = f"  train_loss={train_loss:.4f}"
        if val_loss is not None:
            status += f" | val_loss={val_loss:.4f}"
        else:
            status += " | val_loss=NA"
        print(status, flush=True)

    save_training_artifacts(
        output_dir,
        train_losses,
        val_losses,
        model,
        optimizer,
        scheduler,
        tokenizer,
        tokenizer_save_path,
    )
    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
