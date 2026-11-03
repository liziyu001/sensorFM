# 02 · 仓库骨架与 PyTorch 实现设计

> 目标：在写第一行训练代码之前，把"模块边界、张量形状、测试方式、扩展路径"定下来。
> 原则：**形状优先、纯函数优先、单卡代码即多卡代码、先小后大。**
> 当前状态：设计文档，尚未创建 `src/` 下的代码。

---

## 1. 设计原则

1. **形状优先**：每个模块的 docstring 第一行写输入/输出形状。写不出形状，说明还没想清楚。
2. **纯函数优先**：`patchify`、`unpatchify`、`build_artificial_mask`、`masked_mse` 都是无状态函数，可以单独测。
3. **配置驱动**：模型规模、patch 长度、掩码比例、数据量全部放 YAML；代码里不出现魔法数字。
4. **单卡代码即多卡代码**：用 `torchrun` 启动，代码里只判断 `dist.is_initialized()`；不写两套训练循环。
5. **可复现**：seed、数据分片、fold 划分固定；checkpoint 存 model / optimizer / scheduler / step / RNG。
6. **先小后大**：XXS（13.9 万参数）在 CPU 上跑通，再谈 B（1.1 亿）。
7. **不做未理解的抽象**：第一版不引入 Lightning / Hydra / FSDP；用最直白的 PyTorch。

---

## 2. 目标仓库骨架

```
sensorfm-lab/
├── README.md
├── docs/
│   ├── 00-plan.md
│   ├── 01-model-design.md
│   └── 02-code-design.md
├── pyproject.toml
├── configs/
│   ├── debug.yaml              # 极小模型 + 极小数据，用于单测
│   ├── synthetic_xxs.yaml
│   └── synthetic_b.yaml
├── src/sensorfm/
│   ├── __init__.py
│   ├── config.py               # dataclass + YAML 加载
│   ├── data/
│   │   ├── schema.py           # 34 通道定义：名字、单位、范围
│   │   ├── synthetic.py        # 合成数据生成器（含 latent 与标签）
│   │   ├── features.py         # 原始信号 → 分钟级特征（P1 才用）
│   │   ├── windows.py          # 24h 窗口采样
│   │   ├── missing.py          # 缺失机制注入（MCAR / block / modality）
│   │   └── dataset.py          # torch Dataset / IterableDataset
│   ├── model/
│   │   ├── patchify.py         # [B,T,C] ↔ [B,K,C,P]
│   │   ├── pos_enc.py          # 2D 位置编码 + 8 维循环特征
│   │   ├── block.py            # pre-LN Transformer block
│   │   ├── aim.py              # 人工掩码构造
│   │   └── sensorfm.py         # 顶层模型：编码器 + 解码器 + loss mask
│   ├── train/
│   │   ├── loop.py             # 训练循环（AMP、梯度累积、日志）
│   │   ├── checkpoint.py       # 保存/恢复
│   │   └── distributed.py      # DDP 初始化与 sampler
│   └── eval/
│       ├── probe.py            # 冻结编码器 + PCA + 线性头
│       ├── impute.py           # 插补/外推评估
│       └── metrics.py          # AUC、Pearson、MAE、bootstrap CI
├── scripts/
│   ├── make_synthetic.py
│   ├── train.py
│   └── probe.py
├── tests/
│   ├── test_patchify.py
│   ├── test_aim.py
│   └── test_shapes.py
└── notebooks/
    └── 01_reconstruction.ipynb
```

**第一版只实现 7 个文件**（其余留空或后补）：

```
src/sensorfm/data/schema.py
src/sensorfm/data/synthetic.py
src/sensorfm/model/patchify.py
src/sensorfm/model/pos_enc.py
src/sensorfm/model/aim.py
src/sensorfm/model/sensorfm.py
src/sensorfm/train/loop.py
```

---

## 3. 模块 API 与张量形状

约定：

- `B` batch size，`T=1440` 分钟，`C=34` 通道，`P=20` patch 长度，`K=T/P=72`，`N=K*C=2448`，`d` 模型维度。
- 布尔 mask 一律 **True = 缺失/屏蔽**（和 PyTorch 的 `key_padding_mask` 语义一致）。
- 时间维在前，通道维在后：`[B, T, C]`。

### 3.1 `schema.py`

```python
@dataclass(frozen=True)
class ChannelSpec:
    name: str          # "heart_rate"
    group: str         # "cardio" | "spo2" | "sleep" | "motion" | "skin"
    unit: str          # "bpm" | "ms" | "C" | "S" | "steps/min" ...
    valid_min: float
    valid_max: float

CHANNELS: tuple[ChannelSpec, ...]  # 长度 34，顺序固定
```

作用：所有数据管线、归一化、可视化都引用同一份 schema，避免"第 17 列到底是 LF 还是 HF"这种问题。

### 3.2 `patchify.py`

```python
def patchify(x: Tensor, patch_time: int = 20) -> Tensor:
    """[B, T, C] -> [B, K, C, P]"""

def unpatchify(tokens: Tensor, T: int, C: int) -> Tensor:
    """[B, K, C, P] -> [B, T, C]"""

def patchify_mask(mask: Tensor, patch_time: int = 20,
                  rule: str = "any") -> Tensor:
    """[B, T, C] -> [B, K, C]
    rule="any":  patch 内有任意缺失分钟 → token 缺失
    rule="all":  整个 patch 都缺失       → token 缺失
    """
```

**第一版用 `rule="any"`**：简单、保守；代价是丢弃部分观测。等跑通后再试 `rule="all"` + 分钟级注意力掩码。

### 3.3 `pos_enc.py`

```python
class SensorPE(nn.Module):
    """输入 cyclic: [B, K, 8]；输出 PE: [B, K, C, d]"""
    def __init__(self, n_channels: int, n_patches: int, d_model: int):
        self.channel = nn.Embedding(n_channels, d_model // 2)
        # 时间维：d_model//2 - 8 维正弦；剩余 8 维由 cyclic 填充

    def forward(self, cyclic: Tensor) -> Tensor:
        # [B,K,d/2] = concat(sinusoidal[K, d/2-8], cyclic[B,K,8])
        # [B,K,C,d] = concat(channel[C,d/2], time[B,K,d/2])
```

`cyclic` 的 8 维：`sin/cos(minute_of_hour, hour_of_day, day_of_week, day_of_year)`。

### 3.4 `aim.py`

```python
def build_artificial_mask(
    B: int, K: int, C: int,
    *,
    mode_probs: tuple[float, float, float] = (1/3, 1/3, 1/3),
    random_ratio: float = 0.8,
    temporal_ratio: float = 0.5,
    modality_ratio: float = 0.5,
    generator: torch.Generator | None = None,
) -> Tensor:
    """返回 [B, K, C] bool，True = 人工掩码。"""

def combine_masks(inherited: Tensor, artificial: Tensor) -> tuple[Tensor, Tensor]:
    """输入 [B,K,C] bool；返回 (full_mask, loss_mask)
    full_mask = inherited | artificial
    loss_mask = artificial & ~inherited
    """
```

**不变量**（写成测试）：

```
loss_mask & inherited == 0
loss_mask | inherited == full_mask
```

### 3.5 `sensorfm.py`

```python
class SensorFM(nn.Module):
    def forward(self, x: Tensor, inherited: Tensor, cyclic: Tensor):
        """
        x:         [B, T, C] float
        inherited: [B, K, C] bool   True = 真实缺失
        cyclic:    [B, K, 8] float
        returns:
          pred:      [B, K, C, P]
          loss_mask: [B, K, C] bool
        """
```

内部流程：

```
tokens = patchify(x)                    # [B,K,C,P]
e = in_proj(tokens) + pe(cyclic)        # [B,K,C,d]
art = build_artificial_mask(...)        # [B,K,C]
full, loss_mask = combine_masks(inherited, art)
h = encoder(e.flatten(1,2), key_padding_mask=full.flatten(1,2))
z = dec_in(h.view(B,K,C,d))
z = where(full, mask_token, z)          # [B,K,C,d_dec]
pred = out_proj(decoder(z.flatten(1,2))).view(B,K,C,P)
```

### 3.6 `loop.py`

```python
def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg):
    for step, batch in enumerate(loader):
        x, inherited, cyclic = batch
        pred, loss_mask = model(x, inherited, cyclic)
        target = patchify(x)
        loss = masked_mse(pred, target, loss_mask)
        ...
```

### 3.7 `masked_mse`

```python
def masked_mse(pred, target, loss_mask):
    # pred,target: [B,K,C,P]; loss_mask: [B,K,C]
    m = loss_mask.unsqueeze(-1).expand_as(pred)
    se = (pred - target).pow(2)
    return (se * m).sum() / m.sum().clamp_min(1)
```

---

## 4. 关键代码草图

> 这些是**设计草图**，不是最终实现；写进仓库时会补 docstring、类型标注和测试。

### 4.1 patchify / unpatchify

```python
def patchify(x, patch_time=20):
    B, T, C = x.shape
    assert T % patch_time == 0
    K = T // patch_time
    return x.view(B, K, patch_time, C).permute(0, 1, 3, 2).contiguous()

def unpatchify(tokens, T, C):
    B, K, C_, P = tokens.shape
    assert K * P == T and C_ == C
    return tokens.permute(0, 1, 3, 2).contiguous().view(B, T, C)
```

### 4.2 位置编码

```python
def sinusoidal(n, d):
    pos = torch.arange(n).unsqueeze(1).float()
    i = torch.arange(0, d, 2).float()
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

class SensorPE(nn.Module):
    def __init__(self, n_channels, n_patches, d_model):
        super().__init__()
        assert d_model % 2 == 0
        self.channel = nn.Embedding(n_channels, d_model // 2)
        self.register_buffer("time_sin", sinusoidal(n_patches, d_model // 2 - 8))

    def forward(self, cyclic):                 # cyclic: [B,K,8]
        B, K, _ = cyclic.shape
        C = self.channel.num_embeddings
        ch = self.channel.weight.unsqueeze(0).expand(B, -1, -1)
        ch = ch.unsqueeze(1).expand(-1, K, -1, -1)          # [B,K,C,d/2]
        t = self.time_sin[:K].unsqueeze(0).unsqueeze(2)
        t = t.expand(B, -1, C, -1)                           # [B,K,C,d/2-8]
        cyc = cyclic.unsqueeze(2).expand(-1, -1, C, -1)      # [B,K,C,8]
        t = torch.cat([t, cyc], dim=-1)                      # [B,K,C,d/2]
        return torch.cat([ch, t], dim=-1)                    # [B,K,C,d]
```

### 4.3 AIM 掩码

```python
def build_artificial_mask(B, K, C, generator=None,
                          random_ratio=0.8, temporal_ratio=0.5,
                          modality_ratio=0.5):
    art = torch.zeros(B, K, C, dtype=torch.bool)
    modes = torch.randint(0, 3, (B,), generator=generator)
    for b in range(B):
        m = modes[b].item()
        if m == 0:                                    # 随机
            art[b] = torch.rand(K, C, generator=generator) < random_ratio
        elif m == 1:                                  # 时间块
            length = max(1, int(K * temporal_ratio))
            start = torch.randint(0, K - length + 1, (1,), generator=generator).item()
            art[b, start:start + length, :] = True
        else:                                         # 模态块
            n = max(1, int(C * modality_ratio))
            ch = torch.randperm(C, generator=generator)[:n]
            art[b, :, ch] = True
    return art

def combine_masks(inherited, artificial):
    full = inherited | artificial
    loss_mask = artificial & ~inherited
    return full, loss_mask
```

### 4.4 Transformer block

```python
class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(d * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(d * mlp_ratio), d), nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x
```

注意：`key_padding_mask` 里 True 表示"忽略"。如果一个 batch 里某行全是 True，softmax 会 NaN；训练循环里要保证每个样本至少有一个 visible token。

### 4.5 顶层模型

```python
class SensorFM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.patch_time, cfg.d_model)
        self.pe = SensorPE(cfg.n_channels, cfg.n_patches, cfg.d_model)
        self.encoder = nn.ModuleList([
            Block(cfg.d_model, cfg.enc_heads) for _ in range(cfg.enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        self.dec_in = nn.Linear(cfg.d_model, cfg.d_dec)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, cfg.d_dec))
        self.decoder = nn.ModuleList([
            Block(cfg.d_dec, cfg.dec_heads) for _ in range(cfg.dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(cfg.d_dec)
        self.out_proj = nn.Linear(cfg.d_dec, cfg.patch_time)

    def forward(self, x, inherited, cyclic):
        B, T, C = x.shape
        K = T // self.cfg.patch_time

        tokens = patchify(x, self.cfg.patch_time)        # [B,K,C,P]
        e = self.in_proj(tokens) + self.pe(cyclic)       # [B,K,C,d]

        art = build_artificial_mask(B, K, C)
        full, loss_mask = combine_masks(inherited, art)

        h = e.flatten(1, 2)                              # [B,N,d]
        kpm = full.flatten(1, 2)                         # [B,N]
        for blk in self.encoder:
            h = blk(h, key_padding_mask=kpm)
        h = self.enc_norm(h).view(B, K, C, -1)

        z = self.dec_in(h)
        z = torch.where(full.unsqueeze(-1), self.mask_token, z)
        z = z.flatten(1, 2)
        for blk in self.decoder:
            z = blk(z)
        z = self.dec_norm(z)
        pred = self.out_proj(z).view(B, K, C, self.cfg.patch_time)
        return pred, loss_mask
```

---

## 5. 训练循环（第一版）

```python
def train(cfg):
    model = SensorFM(cfg.model).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.max_steps,
        pct_start=0.05,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    for step, batch in enumerate(loader):
        x, inherited, cyclic = [t.to(cfg.device) for t in batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp):
            pred, loss_mask = model(x, inherited, cyclic)
            loss = masked_mse(pred, patchify(x), loss_mask)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % cfg.log_every == 0:
            log(step, loss, loss_mask.float().mean(), throughput())
```

第一版不做：梯度累积、DDP、checkpoint 分片、数据流式加载。这些在 P2 再加。

---

## 6. 测试策略

| 测试 | 断言 |
|---|---|
| `test_patchify_roundtrip` | `unpatchify(patchify(x)) == x` |
| `test_patchify_shapes` | `[B,1440,34] -> [B,72,34,20]` |
| `test_mask_invariants` | `loss_mask & inherited == 0`；`loss_mask \| inherited == full` |
| `test_forward_shapes` | `pred.shape == [B,72,34,20]`；loss 是标量 |
| `test_no_all_masked_row` | 每个样本至少 1 个 visible token |
| `test_determinism` | 同一 seed → 同一掩码 |
| `test_overfit_one_batch` | 8 个样本、200 步、loss < 1e-3 |

**过拟合测试是最重要的**：如果模型连 8 个样本都记不住，说明代码有 bug（形状错、mask 错、loss 错），不用往下走。

---

## 7. 从单卡到集群的扩展路径

| 版本 | 能力 | 关键改动 |
|---|---|---|
| v0 | 单进程、合成数据、内存 | 本文的 7 个文件 |
| v1 | 单机多卡 DDP | `torchrun`、`DistributedSampler`、`all_reduce` loss |
| v2 | 流式大数据 | Parquet/WebDataset 分片、`IterableDataset`、预取 |
| v3 | 大规模稳定训练 | checkpoint 分片、故障恢复、MFU 监控、LR 扫描 |

**每一步都要能回退**：v1 的代码在单卡上也要能跑（`world_size=1`）。

---

## 8. 现在不做什么

- 不做 FSDP / DeepSpeed / 张量并行：110M 模型不需要。
- 不做自定义 CUDA / FlashAttention：先用 PyTorch 内置。
- 不做 agentic head search：和模型结构无关，后补。
- 不做健康 Agent / 医生评估：属于产品层。
- 不做真实数据下载：先把合成数据跑通。
- 不做超参搜索：第一版只验证"能学"。

---

## 9. 第一版代码的验收标准

1. `pytest` 全绿（6 个测试）。
2. 8 个合成样本、200 步内 loss < 1e-3。
3. 一张重建图：输入 vs 输出，肉眼可见 circadian 和活动峰。
4. 一个线性探针：从冻结 embedding 预测合成 latent，R² > 0.5。
5. 一次消融：AIM vs "先插补再 MAE"，在 50% 缺失下 AIM 的插补 MSE 更低。

达到这 5 条，才算"结构跑起来了"。在此之前，不谈效果。
