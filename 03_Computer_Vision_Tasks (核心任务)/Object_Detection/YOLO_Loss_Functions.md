# YOLO 损失函数深度解析

## 概述

损失函数是 YOLO 性能的关键。从 v1 的简单 MSE，到 v4+ 的 CIoU 损失，体现了目标检测优化思想的进步。

## 损失函数的三个部分

目标检测损失通常包含三个部分：

$$L_{\text{total}} = L_{\text{coord}} + L_{\text{conf}} + L_{\text{class}}$$

### 1. 坐标损失 $L_{\text{coord}}$
预测边界框坐标的准确性。

### 2. 置信度损失 $L_{\text{conf}}$
判断网格中是否存在目标。

### 3. 分类损失 $L_{\text{class}}$
如果有目标，预测其类别。

---

## 演变历程

### v1: 原始 MSE 损失

所有三个部分都使用 L2 损失（MSE）：

$$L = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2\right]$$
$$+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2\right]$$
$$+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2$$
$$+ \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2$$
$$+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{\text{obj}} \sum_{c} (p_i(c) - \hat{p}_i(c))^2$$

**问题**：
- 小框和大框的坐标误差同等权重（小框影响更大）
- 使用 $\sqrt{w}, \sqrt{h}$ 有些临时抱佛脚

### v2/v3: 改进的权重平衡

加入权重项，对不同样本类型区分对待：

$$L_{\text{coord}} = \lambda_{\text{coord}} \sum_{i,j} \mathbb{1}_{ij}^{\text{obj}} (L_{\text{box}})$$
$$L_{\text{conf}} = \sum_{i,j} \mathbb{1}_{ij}^{\text{obj}} (\text{BCE}) + \lambda_{\text{noobj}} \sum_{i,j} \mathbb{1}_{ij}^{\text{noobj}} (\text{BCE})$$

其中 BCE = 二元交叉熵

**改进**：
- 使用二元交叉熵替代 MSE
- 不平衡样本加权（$\lambda_{\text{noobj}} \approx 0.5$，背景框远多于目标框）

---

## 现代损失函数

### IoU Loss

**基本思想**：直接优化 IoU，而非坐标

$$L_{\text{IoU}} = 1 - \text{IoU}(B_{\text{pred}}, B_{\text{gt}})$$

其中：
$$\text{IoU} = \frac{|B_{\text{pred}} \cap B_{\text{gt}}|}{|B_{\text{pred}} \cup B_{\text{gt}}|}$$

**优势**：
- 与评估指标对齐（mAP 基于 IoU）
- 尺度不变性
- 简洁明了

**劣势**：
- 不重叠时梯度为 0（无法优化）
- 优化速度可能变慢

### GIoU Loss（Generalized IoU）

改进 IoU Loss 的梯度问题。

$$\text{GIoU} = \text{IoU} - \frac{|C - (B_{\text{pred}} \cup B_{\text{gt}})|}{|C|}$$

其中 $C$ 是同时包含两个框的最小矩形。

$$L_{\text{GIoU}} = 1 - \text{GIoU}$$

**优势**：
- 即使不重叠也有梯度
- 考虑两个框的相对位置
- 收敛更快

**可视化**：
```
不重叠情况：
┌─────────────┐
│ 预测框       │  ┌─────────┐
│             │  │ 真值框   │
└─────────────┘  └─────────┘

GIoU 考虑最小包含矩形的关系
```

### DIoU Loss（Distance IoU）

加入中心距离约束。

$$L_{\text{DIoU}} = 1 - \text{IoU} + \frac{\rho^2(b_{\text{pred}}, b_{\text{gt}})}{c^2}$$

其中：
- $\rho$ 是两个框中心的欧氏距离
- $c$ 是最小包含矩形的对角线长度

**优势**：
- 加快收敛速度
- 对轴对齐更好

### CIoU Loss（Complete IoU）⭐ 推荐

完整结合了 IoU、距离、纵横比信息。

$$L_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{\text{gt}})}{c^2} + \alpha v$$

其中：
$$v = \frac{4}{\pi^2} (\arctan \frac{w^{\text{gt}}}{h^{\text{gt}}} - \arctan \frac{w}{h})^2$$

$$\alpha = \frac{v}{(1-\text{IoU}) + v}$$

**完整的损失设计**：
- **第一项** $(1-\text{IoU})$：直接优化重叠面积
- **第二项** $\frac{\rho^2}{c^2}$：拉近中心点、收缩包含框
- **第三项** $\alpha v$：纠正纵横比

**优势**：
- 收敛最快
- 精度最高
- YOLOv4+ 采用

---

## 类别不平衡的处理

### 问题
检测数据集中通常：
- 背景（无目标）：90%
- 前景（有目标）：10%
- 各类别间也有严重不平衡

### 解决方案

#### 1. 加权损失
对稀有类别提高权重：

$$L_{\text{weighted}} = \sum_c w_c \cdot L_c$$

其中 $w_c = \frac{1}{p_c}$，$p_c$ 是类别 $c$ 的比例

#### 2. Focal Loss

由 RetinaNet 提出，特别针对前景-背景不平衡。

$$\text{FL}(p_t) = -\alpha_t (1-p_t)^{\gamma} \log(p_t)$$

其中：
- $p_t$ 是真实类别的预测概率
- $\gamma$ 是聚焦参数（通常 $\gamma = 2$）
- $(1-p_t)^{\gamma}$ 是聚焦项

**工作原理**：
- 对于 $p_t \approx 1$（简单样本），聚焦项接近 0，损失很小
- 对于 $p_t \approx 0$（困难样本），聚焦项接近 1，损失很大
- 自动降低简单样本的权重，集中学习困难样本

**应用**：
```python
import torch.nn.functional as F

def focal_loss(pred, target, gamma=2, alpha=0.25):
    # pred: (N, C)
    # target: (N,)
    ce = F.cross_entropy(pred, target, reduction='none')
    p_t = torch.exp(-ce)
    focal = (1 - p_t) ** gamma
    loss = alpha * focal * ce
    return loss.mean()
```

#### 3. OHEM（Online Hard Example Mining）

在线困难样本挖掘：
1. 计算所有样本的损失
2. 只保留困难样本（损失最大的 $k$ 个）
3. 只对这些样本计算反向传播

```python
# 伪代码
all_losses = compute_losses(predictions, targets)
k = batch_size // 4  # 保留 25% 困难样本
hard_indices = torch.topk(all_losses, k)[1]
loss = all_losses[hard_indices].mean()
```

---

## 位置损失的选择对比

| 损失类型 | 特点 | 适用场景 | 收敛速度 |
|---------|------|--------|--------|
| **L1/L2** | 简单，梯度平稳 | 小模型 | 慢 |
| **IoU Loss** | 与指标对齐 | 基准 | 中等 |
| **GIoU** | 改进梯度 | 中等模型 | 较快 |
| **DIoU** | 加入距离 | 复杂背景 | 快 |
| **CIoU** ⭐ | 最完整 | 推荐 | 最快 |

---

## 完整的 YOLO v4 损失函数

### 数学公式
$$L = L_{\text{box}} + L_{\text{obj}} + L_{\text{cls}}$$

### 实现示例

```python
import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def compute_iou(self, box1, box2, eps=1e-7):
        """计算 IoU，box 格式为 [..., 4] = [x1, y1, x2, y2]"""
        x1 = torch.max(box1[..., 0], box2[..., 0])
        y1 = torch.max(box1[..., 1], box2[..., 1])
        x2 = torch.min(box1[..., 2], box2[..., 2])
        y2 = torch.min(box1[..., 3], box2[..., 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * \
                (box1[..., 3] - box1[..., 1]).clamp(min=0)
        area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * \
                (box2[..., 3] - box2[..., 1]).clamp(min=0)
        union = area1 + area2 - inter + eps
        return inter / union
    
    def compute_ciou(self, box1, box2, eps=1e-7):
        """计算 CIoU，box 格式为 [..., 4] = [x1, y1, x2, y2]"""
        iou = self.compute_iou(box1, box2, eps)

        # 中心点距离
        b1_x = (box1[..., 0] + box1[..., 2]) / 2
        b1_y = (box1[..., 1] + box1[..., 3]) / 2
        b2_x = (box2[..., 0] + box2[..., 2]) / 2
        b2_y = (box2[..., 1] + box2[..., 3]) / 2
        rho2 = (b1_x - b2_x) ** 2 + (b1_y - b2_y) ** 2

        # 最小外接矩形对角线
        c_x1 = torch.min(box1[..., 0], box2[..., 0])
        c_y1 = torch.min(box1[..., 1], box2[..., 1])
        c_x2 = torch.max(box1[..., 2], box2[..., 2])
        c_y2 = torch.max(box1[..., 3], box2[..., 3])
        c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

        # 纵横比一致性
        w1 = (box1[..., 2] - box1[..., 0]).clamp(min=eps)
        h1 = (box1[..., 3] - box1[..., 1]).clamp(min=eps)
        w2 = (box2[..., 2] - box2[..., 0]).clamp(min=eps)
        h2 = (box2[..., 3] - box2[..., 1]).clamp(min=eps)
        v = (4 / torch.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2

        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        return iou - rho2 / c2 - alpha * v

    def xywh_to_xyxy(self, boxes):
        """[cx, cy, w, h] 转 [x1, y1, x2, y2]"""
        xy = boxes[..., :2]
        wh = boxes[..., 2:4].clamp(min=1e-7)
        half_wh = wh / 2
        return torch.cat([xy - half_wh, xy + half_wh], dim=-1)
    
    def forward(self, predictions, targets):
        """
        predictions: (B, S, S, 5+C) - [x, y, w, h, conf, class_probs]
        targets: (B, S, S, 5+C)
        """
        batch_size = predictions.size(0)
        
        # 分离预测
        pred_xy = predictions[..., :2]
        pred_wh = predictions[..., 2:4]
        pred_conf = predictions[..., 4]
        pred_class = predictions[..., 5:]
        
        # 分离真值
        target_xy = targets[..., :2]
        target_wh = targets[..., 2:4]
        target_conf = targets[..., 4]
        target_class = targets[..., 5:]
        
        # 是否有目标的掩码
        obj_mask = (target_conf > 0).float()
        noobj_mask = 1 - obj_mask
        
        # 1. 坐标损失（CIoU）
        pred_box = self.xywh_to_xyxy(torch.cat([pred_xy, pred_wh], dim=-1))
        target_box = self.xywh_to_xyxy(torch.cat([target_xy, target_wh], dim=-1))
        ciou = self.compute_ciou(pred_box, target_box)
        loss_coord = torch.sum((1 - ciou) * obj_mask)
        
        # 2. 置信度损失
        loss_obj = torch.sum((pred_conf - target_conf) ** 2 * obj_mask)
        loss_noobj = self.lambda_noobj * torch.sum((pred_conf) ** 2 * noobj_mask)
        loss_conf = loss_obj + loss_noobj
        
        # 3. 分类损失
        loss_class = torch.sum((pred_class - target_class) ** 2 * obj_mask.unsqueeze(-1))
        
        # 总损失
        loss = self.lambda_coord * loss_coord + loss_conf + loss_class
        loss = loss / batch_size
        
        return loss
```

---

## 调试技巧

### 问题 1：损失函数不下降
- 检查学习率是否过大
- 检查数据是否标准化
- 确认损失公式的权重系数设置

### 问题 2：模型过拟合
- 增大正则化系数
- 使用更强的数据增强
- 增加 Dropout

### 问题 3：某个类别性能差
- 增大该类别的权重
- 检查数据不平衡
- 考虑使用 Focal Loss

---

## 参考实现

### PyTorch 中的常用模块
```python
# 交叉熵
loss = nn.CrossEntropyLoss(weight=class_weights)

# IoU 损失（第三方库）
from losses import IoULoss, DIoULoss, CIoULoss

# Focal Loss
from focal_loss import FocalLoss
```

---

## 关键链接

- [[./YOLO完整深度解析|YOLO 完整解析]]
- [[../../02_Deep_Learning_Core (神经网络核心)/Training_Techniques|训练技巧]]
- [[../../../02_Deep_Learning_Core (神经网络核心)/Training_Techniques|正则化与优化]]

---

**学习建议**：理解每种损失函数的数学原理，通过实验对比不同损失在你的数据集上的效果。CIoU Loss 是目前最推荐的选择。

