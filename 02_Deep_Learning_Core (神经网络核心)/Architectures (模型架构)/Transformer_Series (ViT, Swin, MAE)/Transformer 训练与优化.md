# Transformer 训练与优化

## 概述

Transformer 类模型包括 NLP Transformer、Vision Transformer、Swin Transformer、DETR、MAE 和多模态模型。它们通常具有弱归纳偏置、参数量大、层归一化结构明显、对学习率和正则化较敏感等特点。因此，Transformer 的训练配方通常与经典 CNN 不同：更常用 AdamW、warmup、cosine decay、drop path、layer-wise learning rate decay 和较强数据增强。

相关架构基础：[[./Vision Transformer|Vision Transformer]]

## 为什么 Transformer 训练更敏感

### 弱视觉归纳偏置

CNN 天然具有局部连接、权重共享和平移等变性。ViT 将图像切成 patch 后作为 token 处理，模型需要从数据中学习局部纹理和空间关系。结果是：

- 小数据集从零训练容易过拟合。
- 更依赖大规模预训练。
- 数据增强、正则化和学习率调度更重要。

### 注意力的全局交互

Self-Attention 计算：

$$\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

任意 token 都可能影响其他 token。全局交互能力强，但训练中也容易出现：

- 注意力分布早期不稳定。
- 大学习率破坏预训练表征。
- 长序列导致显存和计算开销快速增长。

### LayerNorm 与残差结构

Transformer 由残差连接、LayerNorm、Attention 和 MLP 堆叠。常见结构：

```text
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

这种 Pre-LN 结构比 Post-LN 更稳定，尤其适合深层 Transformer。

## AdamW

### Adam 与 AdamW 的区别

Adam 的更新：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

如果直接在 Adam 的梯度中加入 L2 正则项：

$$g_t' = g_t + \lambda \theta_t$$

这个正则项也会被 Adam 的自适应二阶矩缩放，导致它不再等价于传统意义上的权重衰减。

AdamW 的核心是将权重衰减从梯度中解耦：

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta \lambda \theta_t$$

直觉：

- Adam 负责沿梯度方向更新。
- Weight decay 单独负责让权重变小。
- 正则化强度更可控。

### 常见参数

| 参数 | 常见范围 | 说明 |
|------|----------|------|
| learning rate | `1e-4` 到 `5e-4` | 从头训练通常更大，微调通常更小 |
| betas | `(0.9, 0.999)` | 稳定默认值 |
| betas | `(0.9, 0.95)` | 大模型训练中也常见 |
| eps | `1e-8` | 数值稳定项 |
| weight decay | `0.01` 到 `0.1` | ViT 常用较大 weight decay |

### 不做权重衰减的参数

通常不对以下参数做 weight decay：

- bias。
- LayerNorm 的 scale 和 bias。
- BatchNorm / GroupNorm 参数。
- 位置编码有时也不做衰减，具体取决于代码库惯例。

PyTorch 参数分组示例：

```python
decay = []
no_decay = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if name.endswith("bias") or "norm" in name.lower() or "ln" in name.lower():
        no_decay.append(param)
    else:
        decay.append(param)

param_groups = [
    {"params": decay, "weight_decay": 0.05},
    {"params": no_decay, "weight_decay": 0.0},
]

optimizer = torch.optim.AdamW(
    param_groups,
    lr=5e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
)
```

## Warmup

### 为什么需要 Warmup

训练初期，参数和优化器动量状态尚未稳定。对于 Transformer，直接使用较大学习率容易导致：

- loss 突然增大。
- attention logits 过大。
- 梯度爆炸。
- 预训练权重被破坏。
- 混合精度训练出现 overflow。

Warmup 让学习率从 0 或很小值逐步升到目标学习率：

$$\eta_t = \eta_{\max}\frac{t}{T_{warmup}}$$

### Warmup 设置

常见选择：

- 按 epoch：前 5 到 20 个 epoch warmup。
- 按 step：总训练步数的 1% 到 10%。
- 大 batch 或大模型需要更长 warmup。
- 微调时 warmup 可以短一些，但通常仍建议保留。

### Warmup + Cosine

训练 Transformer 最常见调度：

```text
warmup 阶段：线性升高
主体阶段：余弦下降到最小学习率
```

学习率形式：

$$\eta_t =
\begin{cases}
\eta_{\max}\frac{t}{T_w}, & t < T_w \\
\eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1+\cos\left(\pi\frac{t-T_w}{T-T_w}\right)\right), & t \geq T_w
\end{cases}
$$

PyTorch 示例：

```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_ratio = min_lr / base_lr
    return min_ratio + (1.0 - min_ratio) * cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

## 正则化

### Dropout

Transformer 中常见 dropout 位置：

- Attention probability dropout。
- Projection dropout。
- MLP dropout。
- Embedding dropout。

小数据集微调时，适当 dropout 有帮助；大规模预训练或强增强下，dropout 可以很小甚至为 0。

### DropPath / Stochastic Depth

DropPath 随机丢弃残差分支：

$$x_{l+1} = x_l + m_l F_l(x_l)$$

其中 $m_l$ 是随机 mask。深层 ViT、Swin、ConvNeXt 中非常常见。

实践规律：

- 模型越深，DropPath 可越大。
- 小模型通常 `0.0` 到 `0.1`。
- Base 模型常见 `0.1` 到 `0.2`。
- Large/Huge 模型可更高，但需要验证。

### Label Smoothing

将 hard label 变成 soft label：

$$y' = (1-\epsilon)y + \frac{\epsilon}{C}$$

作用：

- 降低过度自信。
- 改善泛化。
- 与 Mixup/CutMix 搭配时需要调小强度。

### Mixup 与 CutMix

ViT 在图像分类中常配合强增强：

```text
RandAugment + Mixup + CutMix + Random Erasing + Label Smoothing
```

原因：

- ViT 归纳偏置弱，容易记忆训练样本。
- 混合增强迫使模型学习更平滑的决策边界。
- 大规模数据上能显著改善泛化。

风险：

- 小数据集上过强增强可能欠拟合。
- 检测和分割中使用混合增强要严格同步标签。

## Layer-wise Learning Rate Decay

### 动机

微调预训练 Transformer 时，不同层承载的信息不同：

- 低层保留基础纹理、边缘、局部模式。
- 中层表示结构组合。
- 高层更贴近任务语义。

如果所有层使用同一学习率，可能破坏低层通用表示。Layer-wise LR Decay 让越靠近输入的层学习率越小。

公式：

$$\eta_l = \eta_{base} \cdot \alpha^{L-l}$$

其中：

- $L$ 是总层数。
- $l$ 是当前层编号。
- $\alpha$ 通常在 `0.65` 到 `0.9`。

### 示例

```text
head lr = 1e-3
block 11 lr = 8e-4
block 10 lr = 7e-4
...
block 0 lr = 1e-4
patch embed lr = 8e-5
```

适用：

- ViT 微调。
- Swin 微调。
- MAE 预训练模型迁移。
- 下游数据集小于预训练数据集很多的场景。

## 梯度裁剪

Transformer 训练中常使用梯度范数裁剪：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

作用：

- 控制梯度爆炸。
- 稳定 warmup 初期。
- 配合混合精度减少 overflow。

常见范围：

- `max_norm = 1.0` 是稳定起点。
- 大模型或较大学习率可尝试 `0.5`。
- 如果裁剪过于频繁，说明学习率可能过大。

## EMA

EMA 即指数滑动平均模型参数：

$$\theta_{EMA} \leftarrow \alpha \theta_{EMA} + (1-\alpha)\theta$$

作用：

- 平滑训练震荡。
- 提高验证集稳定性。
- 对检测、分割、生成模型常有帮助。

注意：

- EMA 模型通常只用于验证和推理。
- 训练恢复时需要同时保存 EMA 权重。
- decay 过大时，早期更新响应慢。

## 混合精度

Transformer 中大量矩阵乘法适合混合精度：

- FP16：吞吐高，但更容易溢出。
- BF16：数值范围接近 FP32，通常更稳定。

训练建议：

- 支持 BF16 的硬件优先用 BF16。
- FP16 训练使用 GradScaler。
- 出现 NaN 时先降低学习率，再检查 loss scale 和输入数据。

## Vision Transformer 分类训练配方

### 从零训练

```text
model: ViT-B/16
optimizer: AdamW
lr: 5e-4
weight_decay: 0.05
betas: (0.9, 0.999)
batch_size: 512 or larger if possible
scheduler: warmup + cosine decay
warmup_epochs: 5 to 20
epochs: 300 or more
augmentation: RandAugment, Mixup, CutMix, Random Erasing
regularization: label smoothing, drop path
```

说明：

- 从零训练 ViT 需要较多数据和较长训练。
- 小数据上优先使用预训练，而不是从零训练。

### 微调预训练 ViT

```text
optimizer: AdamW
lr: 1e-5 to 5e-4
weight_decay: 0.01 to 0.05
scheduler: warmup + cosine decay
warmup: 1% to 5% total steps
augmentation: weaker than pretraining
layer-wise lr decay: 0.65 to 0.9
drop path: keep or slightly reduce
```

建议：

- 数据少时降低学习率。
- 新分类头学习率可大于 backbone。
- 预训练分辨率和微调分辨率不同时，需要插值位置编码。

## Swin Transformer 训练要点

Swin 的层级结构更接近 CNN，适合检测和分割。

训练特点：

- 使用窗口注意力降低计算量。
- shifted window 提供跨窗口连接。
- 产生多尺度特征，适配 FPN。

常见配置：

```text
optimizer: AdamW
lr: 1e-4 to 5e-4
weight_decay: 0.05
scheduler: warmup + cosine
drop_path: 随模型深度增加
augmentation: 与 ViT 分类类似
```

检测/分割微调：

- backbone 学习率小于 head。
- 使用多尺度训练。
- batch size 小时注意 normalization。
- 权重衰减不作用于 norm 层。

## MAE 预训练与微调

### MAE 预训练

MAE 随机遮挡大量 patch，只编码可见 patch，并让 decoder 重建像素。

特点：

- 遮挡比例高，常见 75%。
- encoder 只处理可见 token，训练效率高。
- 学到的表示适合迁移到下游任务。

预训练关注：

- 足够训练时长。
- 大规模数据。
- 合理 mask ratio。
- 重建目标归一化。

### MAE 微调

微调时常见设置：

- 使用 AdamW。
- 使用 layer-wise lr decay。
- 使用较小 warmup。
- 对位置编码插值。
- 下游分类使用适度增强。

## DETR 类模型训练

DETR 使用 Transformer decoder 和 Hungarian matching，将检测建模为集合预测。

训练特点：

- 收敛慢。
- 对学习率和 backbone/head 分组敏感。
- 原始 DETR 通常需要长训练。
- Deformable DETR 通过稀疏采样加速收敛。

常见策略：

```text
backbone lr: 1e-5
transformer/head lr: 1e-4
optimizer: AdamW
weight_decay: 1e-4
scheduler: step decay or cosine
gradient clipping: 0.1 or 1.0
```

检测相关：[[../../../03_Computer_Vision_Tasks (核心任务)/Object_Detection/Vision_Transformer_in_Object_Detection|Vision Transformer 在目标检测中的应用]]

## 位置编码插值

ViT 的位置编码长度与 patch 数有关。如果预训练尺寸和微调尺寸不同，需要插值：

```text
pretrain: 224x224, patch 16 -> 14x14 tokens
finetune: 384x384, patch 16 -> 24x24 tokens
```

处理方式：

1. 将位置编码从序列还原为二维网格。
2. 使用双三次插值到新网格大小。
3. 再展平成序列。
4. class token 的位置编码单独保留。

如果不插值，模型无法直接加载位置编码，或空间位置信息错位。

## 大 Batch 训练

Transformer 常受益于较大 batch，但过大 batch 会降低梯度噪声，可能影响泛化。

常见做法：

- 增大 batch 时使用 warmup。
- 学习率按线性或平方根规则缩放。
- 使用 gradient accumulation 模拟大 batch。
- 监控验证指标而不是只看训练 loss。

有效 batch：

```text
effective_batch = batch_per_gpu * num_gpus * accumulation_steps
```

## 训练稳定性排查

### Loss 早期发散

优先检查：

- 学习率过大。
- warmup 太短。
- weight decay 过大。
- 梯度裁剪未开启。
- 混合精度 overflow。
- 数据增强过强。

处理：

- 将学习率减半或降到三分之一。
- 增加 warmup steps。
- 开启 `clip_grad_norm_`。
- 暂时关闭 Mixup/CutMix。
- 使用 BF16 或关闭 AMP 验证。

### 微调效果不如线性探测

可能原因：

- 学习率过大，破坏预训练特征。
- 数据太少，全量微调过拟合。
- layer-wise lr decay 未设置。
- 增强和预训练分布不匹配。

处理：

- 冻结低层或降低 backbone lr。
- 增加 weight decay。
- 使用更弱增强。
- 保存并比较不同 epoch 的 checkpoint。

### 验证指标后期下降

可能原因：

- 过拟合。
- 学习率衰减不够。
- 强增强后期仍然过强。
- weight decay 太小。

处理：

- 提前停止。
- 增大 weight decay。
- 后期降低增强。
- 使用 EMA。

## 常见超参数起点

| 场景 | 优化器 | 学习率 | Weight Decay | 调度 | 正则化 |
|------|--------|--------|--------------|------|--------|
| ViT 分类微调 | AdamW | `5e-5` 到 `5e-4` | `0.01` 到 `0.05` | warmup + cosine | drop path, label smoothing |
| ViT 从零训练 | AdamW | `5e-4` | `0.05` 到 `0.1` | warmup + cosine | Mixup, CutMix, RandAugment |
| Swin 检测微调 | AdamW | `1e-4` | `0.05` | warmup + step/cosine | multi-scale, drop path |
| DETR | AdamW | head `1e-4`, backbone `1e-5` | `1e-4` | step/cosine | grad clipping |
| MAE 微调 | AdamW | `1e-4` 到 `1e-3` | `0.05` | warmup + cosine | layer-wise lr decay |

## 与最优化理论的连接

| 训练做法 | 优化含义 |
|----------|----------|
| AdamW | 自适应一阶优化，解耦权重衰减 |
| Warmup | 初期限制步长，降低不稳定更新 |
| Cosine Decay | 后期减小步长，提升收敛精度 |
| Gradient Clipping | 限制更新方向的范数 |
| DropPath | 对网络路径做随机正则 |
| Layer-wise LR Decay | 对不同层使用不同信任半径 |
| EMA | 对参数轨迹做指数平滑 |

## 关键链接

- [[../../../01_Foundations (底层基石)/Math_for_CV (线性代数、概率论、最优化)/最优化|最优化理论]]
- [[./Vision Transformer|Vision Transformer]]
- [[../模型架构总览|模型架构总览]]
- [[../../Training_Techniques (数据增强、优化器、正则化、损失函数)/深度学习训练技巧|深度学习训练技巧]]
- [[../../../03_Computer_Vision_Tasks (核心任务)/Object_Detection/Vision_Transformer_in_Object_Detection|Vision Transformer 在目标检测中的应用]]
