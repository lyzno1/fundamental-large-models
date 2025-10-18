## 项目概述

本仓库用于《大模型基础与应用》期中作业，目标是手工搭建一个可训练的小规模 Transformer，并在小数据集上完成验证、实验分析与文档撰写。当前仓库完成了基础脚手架与实验配置，后续将在此基础上补充完整实现与实验结果。

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
- 运行示例训练（使用合成数据校验流程）：
  ```bash
  ./scripts/run.sh configs/base.yaml
  ```

> 说明：目前默认的 `DummyLanguageModelingDataset` 仅用于验证流程与网络结构，请在后续实验阶段替换为真实数据集（如 WikiText-2）。

## 仓库结构

- `configs/`：实验配置文件（默认 `base.yaml` 对应表 3 超参数设定）。
- `docs/`：作业说明与架构文档。
- `results/`：训练曲线与指标输出位置（当前放置 `.gitkeep` 以追踪目录）。
- `scripts/run.sh`：统一的训练入口脚本，封装了环境变量与命令行参数。
- `src/fundamentals_large_models/`：核心源码，包含配置解析、模型组件与训练流程。
- `main.py`：简易入口，等价于执行 `python -m fundamentals_large_models.train`。

## 模块划分

| 模块 | 功能简介 |
| --- | --- |
| `model/attention.py` | 实现缩放点积注意力与多头自注意力。 |
| `model/feed_forward.py` | 实现位置前馈网络。 |
| `model/blocks.py` | 搭建带残差与 LayerNorm 的 Encoder Block。 |
| `model/positional_encoding.py` | 提供经典正弦位置编码。 |
| `model/transformer.py` | 组合得到最小可训练的 Encoder-only Transformer。 |
| `train.py` | 读取配置、构建模型与优化器、执行训练循环。 |

## 下一步计划

- 接入真实小规模语料（WikiText-2 / Tiny Shakespeare 等），完善 `DataLoader` 与 Tokenizer。
- 补充学习率调度、梯度裁剪等训练稳定性技巧，并保存训练曲线至 `results/`。
- 完成消融实验与报告撰写（LaTeX），确保 README 提供完整复现实验流程。
