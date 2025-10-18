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

- 基于 `uv` 初始化的 Python 项目结构，满足课程要求的目录布局，并自带 `data/local_corpus/` 小语料（同步提供 zip 压缩包）。
- `configs/base.yaml` 现已配置轻量训练设定：`seq_len=64`、`batch_size=8`、`warmup+cosine` 调度，适合笔记本快速验证。
- `src/fundamentals_large_models/` 下的核心模块均已实现并接入训练流水线：
  - `model/*.py`：注意力、前馈、Block、位置编码与 Encoder-only Transformer。
  - `data/dataset.py`：可复现 tokenizer、局部文本数据加载与序列切分。
  - `train.py`：读取配置、构造 scheduler、执行训练/验证循环、写出 `metadata.json`、`metrics.json`、`loss_curve.png`、`model.pt`、`tokenizer.json`。
- `scripts/run.sh`：统一入口，默认执行 `python -m fundamentals_large_models.train --config configs/base.yaml`。
- `results/local_encoder/`（运行后生成）：存放训练曲线、模型权重与实验日志，满足代码开源的可复现要求。

## 3. 后续工作路线

1. **扩展实验规模**：在保留本地小语料的基础上，增加 Hugging Face 公开数据集（如 WikiText-2），并在配置中支持切换。
2. **消融与对比实验**：
   - 去除位置编码、变更注意力头数 / FFN 维度；
   - 引入 decoder block，构建 encoder-decoder 版本提升得分上限。
3. **训练细节强化**：补充梯度裁剪可视化、参数统计、断点恢复脚本等高级特性。
4. **结果整理**：在 `results/` 下按实验编号保存曲线与表格，统一命名便于报告引用。
5. **报告撰写**：按照作业结构撰写不少于 5 页的中文 LaTeX 文档，补充数学推导、伪代码与实验分析。

完成以上事项后，即可准备最终提交与课堂展示。
