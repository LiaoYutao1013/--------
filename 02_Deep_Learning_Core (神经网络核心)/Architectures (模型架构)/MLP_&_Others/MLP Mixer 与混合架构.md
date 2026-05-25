# MLP Mixer 与混合架构

MLP-Mixer 证明了视觉模型不一定必须使用卷积或注意力，只要有合适的 token 混合和通道混合，也能学习图像表示。

## MLP-Mixer 结构

```text
Image
  ↓
Patch Embedding
  ↓
Mixer Block × N
  ├─ Token-mixing MLP
  └─ Channel-mixing MLP
  ↓
Classifier
```

## 两类混合

### Token Mixing

沿空间 token 维度混合信息，相当于让不同 patch 之间通信。

### Channel Mixing

沿通道维度混合信息，相当于在每个位置内部重组特征。

## 与 CNN / Transformer 的对比

| 架构 | 空间交互方式 | 优点 | 局限 |
|------|--------------|------|------|
| CNN | 卷积核局部滑动 | 高效、局部归纳偏置强 | 长距离关系需要堆层 |
| Transformer | 注意力全局交互 | 全局建模强 | 计算量高、需要大数据 |
| MLP-Mixer | MLP 混合 token | 结构简单 | 缺少局部归纳偏置 |

## 混合架构

现代视觉架构往往不是纯 CNN 或纯 Transformer，而是混合设计：
- ConvNeXt：用现代训练策略重新设计 CNN。
- MetaFormer：强调 token mixer 的统一框架。
- MobileViT：把轻量卷积和 Transformer 结合到移动端。
- YOLO 系列：CNN 主干 + PAN/FPN + 注意力或轻量模块。

## 学习价值

MLP-Mixer 不一定是工程首选，但它帮助理解一个问题：模型性能来自哪些因素？

- 局部性是否必要？
- 全局交互是否必要？
- 深度、宽度、归一化和训练策略分别贡献多少？
