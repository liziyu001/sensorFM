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

## 仓库骨架

```
src/sensorfm/
├── config.py                 # YAML -> dataclass
├── data/
│   ├── schema.py             # 34 通道合同（名字/单位/范围）
│   ├── synthetic.py          # 合成数据 + 已知 latent + 缺失机制
│   ├── dataset.py            # torch Dataset
│   ├── windows.py            # 滑窗
│   ├── missing.py            # MCAR / 时间块 / 模态块
│   └── features.py           # 真实数据特征提取（P1 待实现）
├── model/
│   ├── patchify.py           # [B,T,C] <-> [B,K,C,P]
│   ├── pos_enc.py            # 2D 位置编码 + 8 维日历特征
│   ├── block.py              # pre-LN Transformer block
│   ├── aim.py                # mask 构造与不变量
│   └── sensorfm.py           # 顶层模型
├── train/
│   ├── loop.py               # 训练 + masked MSE + eval loss
│   ├── checkpoint.py         # save/load
│   └── distributed.py        # DDP 占位
└── eval/
    ├── probe.py              # person aggregate + linear probe
    ├── impute.py             # 插补评估（待实现）
    └── metrics.py            # AUC / Pearson / MAE / bootstrap CI
```

### 安装与运行（uv）

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/train.py --config configs/debug.yaml
uv run python scripts/make_synthetic.py --config configs/debug.yaml --out data/debug.npz
```

`configs/debug.yaml` 是 CPU 可跑的小模型（XXS，8 人 × 4 天，50 步），用来验证形状和"单 batch 过拟合"。
`configs/synthetic_xxs.yaml` 是单卡配置，`configs/synthetic_b.yaml` 是集群配置（110M 参数）。

## 快速背景（一句话）

SensorFM 是一个可穿戴健康基础模型：把 5 个传感器派生的 34 个分钟级特征、24 小时窗口，切成 20 分钟 × 1 通道的 token，用 ViT-1D + 掩码自编码器（MAE）做自监督预训练。核心创新是 **AIM（Adaptive and Inherited Masking）**：把真实世界的数据缺失和人工掩码合并处理，不插补、不丢弃。

## 当前不做的事

- 不训练、不下载大数据集、不搭集群。
- 不追求复现论文数字（数据/权重/代码均未发布）。
- 不写超出理解范围的代码：先能画清楚一张张量形状图，再写模块。

## 下一步

1. 读完 `docs/01-model-design.md` 中列出的论文段落。
2. 用自己的话回答文末的"理解检查"问题。
3. 对照 `docs/02-code-design.md` 和 `src/sensorfm/` 逐个模块读代码。
4. 装好环境后先跑 `uv run pytest`，再跑 `configs/debug.yaml`。
5. 通过"单 batch 过拟合"后，再讨论合成数据的 scaling 与消融。
