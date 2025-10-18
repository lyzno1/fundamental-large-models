# 《大模型基础与应用》期中作业规划

本文档用于梳理课程作业要求、当前仓库内容以及后续实施路线，便于在撰写报告与补全实验时对照执行。

## 1. 作业要求回顾

- **提交形式**：使用 LaTeX 撰写不少于 5 页的中文报告，并提交编译后的 PDF。
- **基础实现**：自行实现 Multi-Head Self-Attention、Position-wise FFN、残差连接与 LayerNorm、位置编码等模块，实现即可获得 60–70 分。
- **拓展目标**：
  - 搭建完整 Transformer（至少包含 encoder block），在小规模文本建模任务上完成训练与消融实验。
  - 提供框架说明、关键代码、实验设置、指标表格与曲线。
  - 若实现 encoder + decoder，可进一步提升分数。
- **代码开源要求**：
  - 目录需包含 `src/`、`requirements.txt`、`README.md`、`scripts/run.sh`、`results/`。
  - README 中写明硬件环境、依赖安装命令与复现实验的完整命令（含随机种子）。
  - `results/` 中提供训练曲线图与实验表格。
- **训练技巧与评估**：鼓励实现学习率调度、梯度裁剪、AdamW、参数统计、模型保存与训练曲线可视化等。

## 2. 仓库当前内容

- 基于 `uv` 初始化的 Python 项目结构，满足课程要求的目录布局。
- `configs/base.yaml` 复刻了课程说明中的基础超参数表，可作为默认实验配置。
- `src/fundamentals_large_models/` 下已完成核心模块脚手架：
  - `model/attention.py`：缩放点积注意力与多头自注意力。
  - `model/feed_forward.py`：位置前馈网络。
  - `model/blocks.py`：带残差与 LayerNorm 的 encoder block。
  - `model/positional_encoding.py`：正弦位置编码。
  - `model/transformer.py`：组合成最小 encoder-only Transformer，并输出 logits。
  - `train.py`：解析配置、构建模型与优化器、运行示例化训练（当前使用合成数据）。
- `scripts/run.sh`：封装训练入口，可直接执行命令 `./scripts/run.sh configs/base.yaml`。

## 3. 后续工作路线

1. **数据集接入**：选择 Hugging Face 小规模语料（如 `wikitext-2`、`tiny_shakespeare`），实现 Tokenizer、数据预处理与 DataLoader。
2. **训练稳定性**：补齐学习率调度、Warmup、梯度裁剪、模型保存与断点恢复。
3. **实验设计**：依据课程要求完成消融实验，例如：
   - 去除位置编码；
   - 修改注意力头数或 FFN 维度；
   - 分别训练 encoder-only 与 encoder-decoder 架构。
4. **结果整理**：使用 Matplotlib 或 Seaborn 在 `results/` 输出损失曲线、指标表格，并在 README/报告中呈现。
5. **报告撰写**：基于代码结构撰写 LaTeX 报告，包含引言、相关工作、模型推导、伪代码、实验和总结等章节。

按以上路线推进，即可从当前脚手架逐步完善到完整的课程作业提交版本。
