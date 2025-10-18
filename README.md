## 项目概述

本仓库用于《大模型基础与应用》期中作业，目标是手工搭建一个可训练的小规模 Transformer，并在小数据集上完成验证、实验分析与文档撰写。当前版本默认使用 Tiny Shakespeare（Hugging Face: `tiny_shakespeare`）作为示例数据集，并在本地缓存拆分后的文本、训练流程与元数据日志，后续可在此基础上扩展实验与报告。

## 快速上手

- 安装 `uv`（已安装可跳过）：参考 <https://docs.astral.sh/uv/>
- 创建并激活虚拟环境：
  ```bash
  uv venv
  source .venv/bin/activate
  ```
- 安装依赖：
  ```bash
  uv pip install -r requirements.txt
  ```
- 下载 Tiny Shakespeare 并拆分为 train / validation / test（首次必跑，需联网）：
  ```bash
  uv run python scripts/prepare_tiny_shakespeare.py
  ```
- 运行示例训练（使用 Tiny Shakespeare 子集，默认配置适合笔记本）：
  ```bash
  ./scripts/run.sh configs/base.yaml
  ```

训练完成后，会在 `results/tiny_shakespeare_encoder/` 生成：

- `metadata.json`：记录实验、数据、模型、优化器与 tokenizer 配置。
- `metrics.json`：逐 epoch 的训练 / 验证 loss。
- `loss_curve.png`：收敛曲线，可直接引用到报告。
- `model.pt`：包含模型、优化器与调度器权重，便于后续继续训练。
- `tokenizer.json`：与当前语料对应的词表。

## 仓库结构

- `configs/`：实验配置文件（默认 `base.yaml` 对应 Tiny Shakespeare 轻量设置）。
- `data/tiny_shakespeare/`：运行 `prepare_tiny_shakespeare.py` 后生成的本地数据（脚本会自动导出 `tiny_shakespeare.zip`，推荐在提交前将其加入仓库）。
- `docs/`：作业说明与架构文档。
- `results/`：训练曲线、模型权重与实验指标的输出目录。
- `scripts/run.sh`：统一的训练入口脚本，封装了环境变量与命令行参数。
- `src/fundamentals_large_models/`：核心源码，包含配置解析、数据流水线、模型组件与训练流程。
- `main.py`：简易入口，等价于执行 `python -m fundamentals_large_models.train`。

## 模块划分

| 模块 | 功能简介 |
| --- | --- |
| `model/attention.py` | 实现缩放点积注意力与多头自注意力。 |
| `model/feed_forward.py` | 实现位置前馈网络。 |
| `model/blocks.py` | 搭建带残差与 LayerNorm 的 Encoder Block。 |
| `model/positional_encoding.py` | 提供经典正弦位置编码。 |
| `model/transformer.py` | 组合得到最小可训练的 Encoder-only Transformer。 |
| `data/dataset.py` | 构建可复现的 tokenizer、加载本地/ HuggingFace 语料并切分序列。 |
| `train.py` | 读取配置、构建模型与优化器、执行训练与评估循环、输出曲线与模型。 |

## 训练特性速览

- 支持 Tiny Shakespeare 快速迭代，默认 `batch_size=16`、`seq_len=128` 适配常见笔记本。
- `AdamW + 梯度裁剪 + warmup + cosine decay`，可根据配置快速切换。
- 每轮自动在 validation split 评估，并把 loss 曲线保存至 `loss_curve.png`。
- 自动写入 `metadata.json`、`metrics.json`、`model.pt`、`tokenizer.json` 等复现实验所需资产。
- Tokenizer 词表支持缓存 / 加载，保证多次运行的一致性。

## 下一步计划

- 根据课程要求扩展到 Encoder-Decoder 结构，并在更复杂任务上验证。
- 设计完整的消融实验（去除位置编码、修改头数/FFN 维度等），补充结果表格。
- 在 `results/` 中整理多组实验（含随机种子），并撰写 LaTeX 报告不少于 5 页。
