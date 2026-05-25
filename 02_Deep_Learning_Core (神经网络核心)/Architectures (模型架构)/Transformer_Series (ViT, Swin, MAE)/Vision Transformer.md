# Vision Transformer

Vision Transformer（ViT）把图像切成 patch，并把每个 patch 当作一个 token 输入 Transformer。它把 NLP 中的序列建模思想迁移到视觉任务。

## 基本流程

```text
Image H×W×C
  ↓
Patchify: P×P patches
  ↓
Linear Projection
  ↓
Position Embedding
  ↓
Transformer Encoder × L
  ↓
Classification / Detection / Segmentation Head
```

## 核心公式

Self-Attention：

$$\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d}})V$$

其中：
- $Q$：当前 token 想查询什么。
- $K$：每个 token 提供什么索引。
- $V$：每个 token 提供什么内容。

## ViT 与 CNN 的区别

| 对比项 | CNN | ViT |
|--------|-----|-----|
| 归纳偏置 | 局部性、平移等变 | 弱归纳偏置，更依赖数据 |
| 感受野 | 层层扩大 | 一开始就全局交互 |
| 数据需求 | 相对较少 | 通常需要大规模预训练 |
| 多尺度能力 | 天然层级结构 | 原始 ViT 较弱，Swin 改进 |

## Swin Transformer

Swin 的核心改进：
- 使用窗口注意力降低计算量。
- 通过 shifted window 跨窗口交互。
- 构造层级特征，适合检测和分割。

## MAE

Masked Autoencoder（MAE）通过遮挡大量 patch，让模型重建图像内容。

直觉：
- 遮挡比例高，任务更难。
- 编码器只处理可见 patch，训练效率高。
- 学到的表示可迁移到分类、检测、分割。

## 与检测任务的关系

Transformer 在检测中常见两种用法：
- 作为 backbone：如 Swin + Mask R-CNN。
- 作为检测头：如 DETR、Deformable DETR。

相关：[[../../../03_Computer_Vision_Tasks (核心任务)/Object_Detection/Vision_Transformer_in_Object_Detection|ViT 在目标检测中的应用]]
