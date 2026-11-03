# SensorFM 学习与研究仓库

本仓库用于**学习 SensorFM 的模型设计**，并为后续的方法复现与规模化训练做准备。

当前阶段：**阅读与设计**。在把模型结构、数据表示、训练目标和评估口径弄清楚之前，不启动实际训练。

## 论文

- **Towards a General Intelligence and Interface for Wearable Health Data**
  - arXiv: [2605.22759](https://arxiv.org/abs/2605.22759)
  - Google Research Blog: [SensorFM](https://research.google/blog/sensorfm-towards-a-general-intelligence-and-interface-for-wearable-health-data/)
  - 本地：`paper.pdf`（不纳入 git）
- 关键前作：**LSM-2: Learning from Incomplete Wearable Sensor Data**（AIM 的出处）
  - arXiv: [2506.05321](https://arxiv.org/abs/2506.05321)

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/00-plan.md](docs/00-plan.md) | 粗略计划：目标、阶段、数据策略、待确认事项（DRAFT） |
| [docs/01-model-design.md](docs/01-model-design.md) | SensorFM 模型解剖：输入、token 化、位置编码、AIM、损失、下游协议 |
| [docs/02-code-design.md](docs/02-code-design.md) | 仓库骨架与 PyTorch 实现设计：模块 API、张量形状、测试、扩展路径 |
| [SensorFM_调研报告.md](SensorFM_调研报告.md) | 第一轮调研：论文结论、可复现性、借鉴价值 |

## 快速背景（一句话）

SensorFM 是一个可穿戴健康基础模型：把 5 个传感器派生的 34 个分钟级特征、24 小时窗口，切成 20 分钟 × 1 通道的 token，用 ViT-1D + 掩码自编码器（MAE）做自监督预训练。核心创新是 **AIM（Adaptive and Inherited Masking）**：把真实世界的数据缺失和人工掩码合并处理，不插补、不丢弃。

## 当前不做的事

- 不训练、不下载大数据集、不搭集群。
- 不追求复现论文数字（数据/权重/代码均未发布）。
- 不写超出理解范围的代码：先能画清楚一张张量形状图，再写模块。

## 下一步

1. 读完 `docs/01-model-design.md` 中列出的论文段落。
2. 用自己的话回答文末的"理解检查"问题。
3. 确认 `docs/02-code-design.md` 里的模块划分与张量形状。
4. 再决定是否开始写第一版代码（合成数据 + XXS 模型）。
