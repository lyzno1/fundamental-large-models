from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class ExperimentConfig:
    """General experiment-level parameters."""

    name: str = "base-transformer"
    seed: int = 42
    output_dir: Path = Path("results")


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""

    dataset_name: str = "wikitext"
    subset: Optional[str] = "wikitext-2-v1"
    data_dir: Optional[Path] = None
    text_field: str = "text"
    max_length: int = 128
    train_split: str = "train"
    val_split: str = "validation"
    test_split: Optional[str] = "test"
    lowercase: bool = True
    limit_examples: Optional[int] = None


@dataclass
class ModelConfig:
    """Transformer model hyper-parameters."""

    vocab_size: int = 50000
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    max_position_embeddings: int = 512
    use_decoder: bool = False


@dataclass
class OptimConfig:
    """Optimization and training loop parameters."""

    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: Sequence[float] = (0.9, 0.98)
    eps: float = 1e-9
    grad_clip: float = 1.0
    warmup_steps: int = 4000
    scheduler: str = "cosine"
    log_interval: int = 100
    eval_interval: int = 500


@dataclass
class TokenizerConfig:
    """Tokenizer configuration for building vocabularies."""

    type: str = "basic"
    vocab_file: Optional[Path] = None
    lowercase: bool = True
    min_freq: int = 1
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
