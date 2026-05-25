# Vision Transformer 在目标检测中的应用

> 完整笔记已维护在：[[../Object_Detection/Vision_Transformer_in_Object_Detection|Vision_Transformer_in_Object_Detection]]

## 快速理解

Vision Transformer 在目标检测中主要解决的是 **全局建模** 问题：CNN 擅长局部纹理和层级特征，Transformer 擅长长距离依赖和目标之间的关系建模。

目标检测里常见的 Transformer 用法有三类：

| 范式 | 代表方法 | 核心思想 |
|------|----------|----------|
| ViT backbone + 检测头 | ViTDet、Swin + Mask R-CNN | 用 Transformer 替代 CNN 骨干 |
| CNN + Transformer neck/head | DETR、Deformable DETR | 用注意力做目标查询和集合预测 |
| 混合结构 | ConvNeXt/Swin/YOLO 混合模块 | CNN 负责局部特征，Transformer 负责全局关系 |

## DETR 的关键变化

传统检测器通常依赖 anchor、候选框、NMS。DETR 把检测看作集合预测：

```text
Image
  ↓
Backbone feature map
  ↓
Transformer Encoder / Decoder
  ↓
Object Queries
  ↓
Class + Box predictions
```

核心特点：
- 使用 object query 表示一组潜在目标
- 使用匈牙利匹配建立预测与真值的一一对应
- 理论上可以减少对 NMS 和手工 anchor 的依赖
- 原始 DETR 收敛慢，小目标效果弱，后续 Deformable DETR 做了改进

## 学习重点

1. 先掌握 [[./YOLO 算法原理全深度解析 (Master Deep Dive)|YOLO]] 的 dense prediction 思路。
2. 再对比 DETR 的 set prediction 思路，理解为什么它不再逐网格/逐 anchor 预测。
3. 重点关注 Deformable Attention：它用少量采样点替代全局注意力，降低计算量并提升小目标表现。

## 相关链接

- [[../Object_Detection/Vision_Transformer_in_Object_Detection|完整 ViT 检测笔记]]
- [[./YOLO_Loss_Functions|YOLO 损失函数]]
- [[./Small_Object_Detection_Optimization|小目标检测优化]]
