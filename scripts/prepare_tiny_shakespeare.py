#!/usr/bin/env python3
"""Download and prepare the Tiny Shakespeare dataset for local experiments."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Tuple

DEFAULT_OUTPUT = Path("data/tiny_shakespeare")
TINY_SHAKESPEARE_SOURCES = [
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tiny_shakespeare/input.txt",
]


def split_text(text: str, val_ratio: float, test_ratio: float) -> Tuple[str, str, str]:
    total = len(text)
    val_len = int(total * val_ratio)
    test_len = int(total * test_ratio)
    train_len = total - val_len - test_len

    train_text = text[:train_len]
    val_text = text[train_len : train_len + val_len]
    test_text = text[train_len + val_len :]
    return train_text, val_text, test_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Tiny Shakespeare dataset from public raw text sources."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    args = parser.parse_args()

    raw_text = None
    last_error: Exception | None = None
    for url in TINY_SHAKESPEARE_SOURCES:
        try:
            print(f"Attempting download from {url}")
            import urllib.request

            with urllib.request.urlopen(url) as response:
                raw_bytes = response.read()
            if not raw_bytes:
                raise RuntimeError("empty response")
            raw_text = raw_bytes.decode("utf-8")
            break
        except Exception as exc:  # pragma: no cover - network failure
            print(f"Failed to download from {url}: {exc}")
            last_error = exc

    if raw_text is None:
        raise RuntimeError("Unable to download Tiny Shakespeare dataset.") from last_error

    train_text, val_text, test_text = split_text(raw_text, args.val_ratio, args.test_ratio)

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
