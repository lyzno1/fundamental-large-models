#!/usr/bin/env python3
"""Download and prepare the Tiny Shakespeare dataset for local experiments."""

from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path
from typing import Tuple

from datasets import load_dataset  # type: ignore

DEFAULT_OUTPUT = Path("data/tiny_shakespeare")


def split_text(text: str, val_ratio: float, test_ratio: float, seed: int) -> Tuple[str, str, str]:
    rng = random.Random(seed)
    total = len(text)
    val_len = int(total * val_ratio)
    test_len = int(total * test_ratio)
    train_len = total - val_len - test_len

    indices = list(range(total))
    rng.shuffle(indices)

    train_idx = set(indices[:train_len])
    val_idx = set(indices[train_len : train_len + val_len])
    test_idx = set(indices[train_len + val_len :])

    train_chars = []
    val_chars = []
    test_chars = []
    for i, ch in enumerate(text):
        if i in train_idx:
            train_chars.append(ch)
        elif i in val_idx:
            val_chars.append(ch)
        else:
            test_chars.append(ch)

    return "".join(train_chars), "".join(val_chars), "".join(test_chars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Tiny Shakespeare dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Downloading tiny_shakespeare via Hugging Face datasets...")
    dataset = load_dataset("tiny_shakespeare")
    raw_text = dataset["train"][0]["text"]

    train_text, val_text, test_text = split_text(raw_text, args.val_ratio, args.test_ratio, args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.txt").write_text(train_text, encoding="utf-8")
    (output_dir / "validation.txt").write_text(val_text, encoding="utf-8")
    (output_dir / "test.txt").write_text(test_text, encoding="utf-8")

    zip_path = output_dir.parent / "tiny_shakespeare.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for split in ["train", "validation", "test"]:
            zf.write(output_dir / f"{split}.txt", arcname=f"{split}.txt")

    print(f"Dataset prepared under {output_dir}")
    print(f"Archive written to {zip_path}")


if __name__ == "__main__":
    main()
