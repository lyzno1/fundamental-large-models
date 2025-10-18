# 实验结果摘要

本节汇总 `results/` 目录下三组实验的验证集表现，可直接用于期中作业报告与复现说明。

| 配置文件 | 关键改动 | 最终验证 Loss | 目录 |
| --- | --- | --- | --- |
| `configs/base.yaml` | 基线：2 层 Encoder、4 头、FFN=512、带位置编码 | 2.48 | `results/tiny_shakespeare_encoder/` |
| `configs/ablation_no_positional_encoding.yaml` | 去掉正弦位置编码 | 4.54 | `results/tiny_shakespeare_no_pos/` |
| `configs/ablation_reduced_capacity.yaml` | 减少注意力头数为 2，FFN=256 | 2.60 | `results/tiny_shakespeare_reduced/` |

> 备注：Loss 为 `metrics.json` 中 `val_loss` 序列的最后一个值。曲线可查看对应目录下的 `loss_curve.png`。

关键观察：
- 移除位置编码后验证损失显著上升，说明 Transformer 无法仅靠自注意力捕获顺序信息。
- 缩减模型容量对性能也有影响，但幅度远小于完全去位置编码，验证了头数及 FFN 维度主要影响表示能力上限。

后续可在报告中进一步配合参数、推理时长或示例输出做定性分析。
