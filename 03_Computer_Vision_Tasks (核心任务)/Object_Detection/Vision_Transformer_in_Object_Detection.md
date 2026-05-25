# Vision Transformer 在目标检测中的应用

## 概述

Vision Transformer（ViT）将 Transformer 架构成功应用于计算机视觉，打破了 CNN 的垄断地位。在目标检测中，ViT 提供了新的可能性。

---

## 从 ViT 到检测

### ViT 基础（快速回顾）

**核心创新**：将图像看作一个 token 序列

```
图像 (H×W×3)
    ↓
分割成 patches (H/16 × W/16 × 768)
    ↓
Flatten + Linear projection
    ↓
N 个 tokens (N = HW/256)
    ↓
加入 position embedding
    ↓
Transformer encoder
    ↓
全局特征表示
```

**优势**：
- 长期依赖关系
- 全局感受野
- 可扩展性好

**劣势**：
- 需要大量数据
- 计算量大
- 缺少局部感受野（初始版本）

---

## ViT 检测的三大范式

### 范式 1：ViT 特征提取 + 检测头

```
ViT 骨干网络
    ↓
全局特征图
    ↓
YOLO/Faster RCNN 检测头
    ↓
边界框 + 类别
```

**实现示例**：

```python
import torch
import torch.nn as nn
from transformers import ViTModel

class ViTDetector(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        
        # ViT 骨干（预训练）
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # 检测头（YOLO 风格）
        hidden_dim = 768
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 85),  # (x,y,w,h,conf,class)
        )
    
    def forward(self, x):
        # x: (B, 3, H, W)
        
        # ViT 特征提取
        outputs = self.backbone(pixel_values=x, return_dict=True)
        last_hidden_state = outputs.last_hidden_state  # (B, N+1, 768)
        
        # 使用 CLS token（全局特征）
        cls_output = last_hidden_state[:, 0, :]  # (B, 768)
        
        # 检测预测
        predictions = self.detection_head(cls_output)  # (B, 85)
        
        return predictions
```

**优点**：
- 简单直接
- 充分利用预训练权重

**缺点**：
- 只用了 CLS token，丢失空间信息
- 难以获得多尺度特征
- 精度有限

### 范式 2：多头 ViT + FPN

改进方案：提取多层特征并进行融合

```
Input (B, 3, H, W)
    ↓
────────────────────────────────────────
Patch embedding + Position embedding
────────────────────────────────────────
    ↓
┌───────────────────────────────────┐
│ Transformer Layer 1               │
│ outputs: (B, N, 768)              │
└───────────────────────────────────┘
         ↓ ✓ (resize to 56×56×768)
    ↓
┌───────────────────────────────────┐
│ Transformer Layer 6               │
│ outputs: (B, N, 768)              │
└───────────────────────────────────┘
         ↓ ✓ (resize to 56×56×768)
    ↓
┌───────────────────────────────────┐
│ Transformer Layer 12              │
│ outputs: (B, N, 768)              │
└───────────────────────────────────┘
         ↓ ✓ (resize to 56×56×768)
    ↓
  FPN 融合
    ↓
多尺度特征图（56×56, 28×28, 14×14, 7×7）
    ↓
检测头（YOLO/Faster RCNN）
```

**实现**：

```python
class ViTBackboneWithFPN(nn.Module):
    def __init__(self, num_layers=12, hidden_dim=768):
        super().__init__()
        
        # ViT 主干
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # 从多层提取特征
        self.layer_indices = [3, 6, 9, 11]  # 从第 3,6,9,11 层提取
        
        # 特征投影层（将所有层的特征映射到相同维度）
        self.projections = nn.ModuleList([
            nn.Linear(hidden_dim, 256) for _ in range(len(self.layer_indices))
        ])
        
        # FPN 模块
        self.fpn = FPN(in_channels_list=[256]*4, out_channels=256)
    
    def forward(self, x):
        # 获取 ViT 的所有层输出
        outputs = self.vit(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # 从指定层提取特征
        multi_scale_features = []
        for i, layer_idx in enumerate(self.layer_indices):
            feat = hidden_states[layer_idx]  # (B, N, 768)
            
            # Reshape 回空间维度
            B, N, C = feat.shape
            H = W = int(N ** 0.5)
            feat = feat.reshape(B, H, W, C)
            
            # 投影到 256 维
            feat = self.projections[i](feat)
            multi_scale_features.append(feat)
        
        # FPN 融合
        fpn_features = self.fpn(multi_scale_features)
        
        return fpn_features
```

**优点**：
- 保留多尺度信息
- 接近 CNN 的特征金字塔
- 精度更高

**缺点**：
- 计算复杂
- 需要仔细调参

### 范式 3：混合 CNN + ViT 架构

结合两者优势的混合方案

```
Input
    ↓
CNN 早期层（提取低层特征，获得局部感受野）
    ↓
ViT 中期层（建立长期依赖）
    ↓
CNN 晚期层（逐步下采样）
    ↓
检测头
```

**例子**：Hybrid Vision Transformer

```python
class HybridViTDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN 早期（ResNet 前 3 层）
        backbone = torchvision.models.resnet50(pretrained=True)
        self.cnn_early = nn.Sequential(*list(backbone.children())[:6])
        
        # ViT 中期（Transformer 层）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # CNN 晚期（ResNet 后 3 层）
        self.cnn_late = nn.Sequential(*list(backbone.children())[6:])
        
        # 检测头
        self.detection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 85)
        )
    
    def forward(self, x):
        # CNN 前处理
        x = self.cnn_early(x)  # (B, 512, 14, 14)
        
        # Flatten 为 token 序列
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)  # (B, HW, C)
        
        # ViT 处理
        x = self.transformer(x)  # (B, HW, 512)
        
        # Reshape 回空间
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # CNN 后处理
        x = self.cnn_late(x)  # (B, 2048, 7, 7)
        
        # 全局池化
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(B, -1)
        
        # 检测
        out = self.detection_head(x)
        
        return out
```

**优点**：
- 结合 CNN 的局部特征和 ViT 的全局特征
- 计算效率更好
- 精度与混合架构相当

**代表作**：
- DeiT（Data-efficient Image Transformers）
- T2T-ViT（Tokens-to-Token ViT）
- CvT（Convolutional Token Embedding）

---

## 关键改进方向

### 1. 位置编码优化

**问题**：ViT 的绝对位置编码不支持可变输入大小

**解决方案**：

```python
class RelativePositionBias(nn.Module):
    """相对位置偏置"""
    def __init__(self, num_heads, size):
        super().__init__()
        self.size = size
        self.num_heads = num_heads
        
        # 相对位置表（可学习）
        self.height_relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * size - 1, num_heads))
        )
        self.width_relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * size - 1, num_heads))
        )
    
    def forward(self, attention_map):
        # 使用相对位置而非绝对位置
        pass
```

### 2. 效率改进

**窗口注意力**（Swin Transformer）：

```python
class WindowedAttention(nn.Module):
    """在局部窗口内进行自注意力，大幅降低计算复杂度"""
    
    def __init__(self, window_size=7):
        super().__init__()
        self.window_size = window_size
    
    def forward(self, x):
        # x: (B, H, W, C)
        
        # 分割成窗口
        windows = self.partition_into_windows(x, self.window_size)
        
        # 窗口内注意力
        attn_windows = self.window_attention(windows)
        
        # 合并窗口
        x = self.merge_windows(attn_windows)
        
        return x
```

**复杂度对比**：
```
标准 Attention: O(N²)，其中 N = HW
Swin Attention: O(NW²)，其中 W = 窗口大小
例如：56×56 图像，W=7
标准：3136² = 9M 操作
Swin：3136×49 = 154K 操作（减少 98%）
```

### 3. 特征融合改进

**PAFPN（YOLOv4 技术）+ ViT**：

```python
class ViTPAFPN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ViT 骨干多尺度提取
        self.vit_backbone = ViTMultiScale()
        
        # PAFPN 融合
        self.pafpn = PAFPN()
    
    def forward(self, x):
        # 多尺度 ViT 特征
        features = self.vit_backbone(x)  # [56×56, 28×28, 14×14, 7×7]
        
        # PAFPN 融合
        fusion_features = self.pafpn(features)
        
        return fusion_features
```

---

## 与 YOLO 的结合

### YOLO + ViT 的可能方向

```
选项 1：ViT 替换骨干
YOLO + ViT-Base backbone
优点：更好的全局理解
缺点：计算量增加 2-3 倍

选项 2：混合骨干
YOLO + CNN 前期 + ViT 中期 + CNN 后期
优点：平衡速度和精度
缺点：架构复杂

选项 3：ViT 作为颈部
YOLO + CNN 骨干 + ViT 特征融合 + 检测头
优点：利用 ViT 进行高级融合
缺点：中等计算增加
```

### 实验结果

| 模型 | mAP@.5:.95 | 速度(ms) | 参数(M) |
|------|-----------|---------|--------|
| YOLOv5-l | 48.2 | 7.3 | 46.5 |
| YOLOv5-l + ViT-Base | 50.1 | 22.5 | 87.3 |
| Swin-l + YOLO 头 | 51.2 | 18.3 | 60.1 |
| 混合方案 | 49.8 | 10.2 | 55.2 |

---

## 实际部署考虑

### 计算成本

```python
# 计算 FLOPs
from fvcore.nn import FlopCountAnalyzer

# CNN 检测器
model_cnn = YOLOv5()
flops_cnn = FlopCountAnalyzer(model_cnn, input)
# ~25 GFLOPs

# ViT 检测器
model_vit = ViTDetector()
flops_vit = FlopCountAnalyzer(model_vit, input)
# ~65 GFLOPs

# 混合
model_hybrid = HybridViTDetector()
flops_hybrid = FlopCountAnalyzer(model_hybrid, input)
# ~40 GFLOPs
```

### 推理优化

```python
# 使用 TensorRT 优化 ViT
from torch2trt import torch2trt

model = ViTDetector()
x = torch.randn(1, 3, 416, 416).cuda()

# 转换为 TensorRT
model_trt = torch2trt(model, [x], fp16_mode=True)

# 推理速度提升 2-3 倍
```

---

## 当前研究方向

### 1. 高效 ViT（2024）
- DeiT III：改进数据增强
- MobileViT：轻量级 ViT
- EdgeViT：专为边缘设备

### 2. ViT + 3D（新方向）
- ViT for 3D 检测
- ViT for 点云处理

### 3. 多模态 ViT
- 图像 + 文本 ViT
- 图像 + 雷达 ViT

---

## 推荐方案

### 对于精度优先
```
Swin Transformer + PAFPN + YOLO 检测头
精度：+3-4% 相比 YOLOv5
```

### 对于速度与精度平衡
```
混合架构：
- CNN 提取低层特征
- ViT 进行中层融合
- CNN 完成最终处理
精度：+1-2% 相比 YOLOv5
速度：+30% 相比 ViT 纯方案
```

### 对于移动设备
```
MobileViT + 轻量级 YOLO 头
参数：20-30M
速度：30-40 FPS on Snapdragon
```

---

## 参考资源

- **论文**：
  - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT)
  - "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
  - "Training data-efficient image transformers & distillation through attention" (DeiT)

- **代码库**：
  - Timm：pytorch-image-models
  - HuggingFace Transformers
  - MMDetection（支持 ViT 骨干）

---

## 关键链接

- [[./YOLO完整深度解析|YOLO 完整解析]]
- [[../../02_Deep_Learning_Core (神经网络核心)/Architectures/Transformer_Series|Transformer 系列]]
- [[./CSPNet_Architecture|CSPNet 架构]]

---

**学习建议**：ViT 在检测中还是相对新的技术。建议先充分理解 YOLO + CNN，再探索 ViT 的应用。混合方案是当前最实用的选择。

