# 小目标检测优化完全指南

## 问题定义

小目标检测的关键挑战：

1. **信息丢失**：小目标像素少，特征不明显
2. **特征混淆**：小目标的特征类似于大目标的纹理
3. **多尺度困难**：难以在多个尺度上都检测好
4. **背景干扰**：高背景噪声导致误检
5. **计算效率**：高分辨率输入增加计算量

---

## 1. 输入分辨率优化

### 提高输入分辨率
YOLO 默认输入 640×640，小目标检测需要更高分辨率。

```python
# YOLOv5 示例
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

# 标准：640×640
results = model('image.jpg', imgsz=640)

# 小目标优化：1280×1280
results = model('image.jpg', imgsz=1280)

# 或者使用矩形推理保持宽高比
results = model('image.jpg', imgsz=1280, rect=True)
```

**权衡**：
- 优点：更多像素表示小目标
- 缺点：内存翻倍，推理时间增加 3-4 倍

### 自适应输入大小
```python
def adaptive_input_size(img_shape, target_objects_size='small'):
    """根据目标大小选择输入分辨率"""
    h, w = img_shape
    
    if target_objects_size == 'small':
        return 1280  # 小目标
    elif target_objects_size == 'medium':
        return 640
    else:  # large
        return 416
```

---

## 2. 特征金字塔改进

### 增加小尺度特征图

标准 YOLO 使用三个尺度（13×13, 26×26, 52×52），需要添加更小的尺度。

```
标准 FPN：
        主干网络
           ↓
    ┌───────┼────────┐
   64×64  128×128  256×256  (YOLOv5 的最小尺度)
    
改进的 FPN for 小目标：
        主干网络
           ↓
    ┌───────┼────────┬──────┐
  64×64  128×128  256×256  512×512  (额外超小尺度)
```

```python
# 自定义增强 FPN
class EnhancedFPN(nn.Module):
    def __init__(self):
        super().__init__()
        # 添加额外的上采样层用于超小目标
        self.extra_upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1),
        ])
    
    def forward(self, features):
        # features: 来自主干网络
        # 生成额外的小尺度特征
        pass
```

### 使用 PAFPN（Path Aggregation FPN）

YOLOv4+ 采用的改进 FPN 结构，特征融合更高效。

---

## 3. Anchor 优化

### 针对小目标的先验框设计

使用聚类得到数据集特定的 anchor：

```python
from kmeans import kmeans, avg_iou
import numpy as np

# 从数据集中聚类 anchor
def generate_anchors_for_small_objects(annotations, k=9):
    """使用 K-means 生成最优 anchor"""
    sizes = []
    for ann in annotations:
        # 提取边界框尺寸
        w = ann['bbox'][2] - ann['bbox'][0]
        h = ann['bbox'][3] - ann['bbox'][1]
        sizes.append([w, h])
    
    sizes = np.array(sizes)
    
    # K-means 聚类
    centroids, avg_iou_val = kmeans(sizes, k=k)
    
    # 按面积排序
    anchors = centroids[np.argsort(centroids[:, 0] * centroids[:, 1])]
    
    return anchors
```

**结果示例**：
```
通用 COCO anchors:
[10, 13], [16, 30], [33, 23], ...

小目标数据集（如无人机）：
[5, 6], [8, 12], [12, 15], ...  (更小)
```

### 无锚设计

新的 anchor-free 方法对小目标更友好：

```python
# FCOS / YOLOv6+ 的无锚方法
# 直接预测：
# - 目标的中心点
# - 距离边界的距离（l, t, r, b）
# 而不是相对于 anchor 的偏移
```

---

## 4. 损失函数优化

### 使用 focal loss 处理类别不平衡

小目标通常比大目标少，导致类别不平衡。

```python
def focal_loss(pred, target, alpha=0.25, gamma=2):
    """Focal Loss：自动给困难样本更高的权重"""
    ce = F.cross_entropy(pred, target, reduction='none')
    p_t = torch.exp(-ce)
    focal = (1 - p_t) ** gamma
    loss = alpha * focal * ce
    return loss.mean()

# 对小目标额外加权
class WeightedFocalLoss(nn.Module):
    def __init__(self, object_size_threshold=32*32):
        self.threshold = object_size_threshold
    
    def forward(self, pred, target, bboxes):
        """加权 Focal Loss：小目标权重更高"""
        base_loss = focal_loss(pred, target)
        
        # 计算目标面积
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        
        # 小目标加权
        weights = torch.where(areas < self.threshold, 
                             torch.ones_like(areas) * 2.0,
                             torch.ones_like(areas))
        
        return (base_loss * weights).mean()
```

### 使用 GIoU 或 CIoU 损失

对于小目标，完整的几何信息至关重要。

[[./YOLO_Loss_Functions|详见损失函数专题]]

---

## 5. 数据增强策略

### Mosaic 增强

特别有效，四张图合成一张大大增加小目标的数量。

```python
def mosaic_9(images, labels):
    """9 张图拼接（3×3）"""
    mosaic_img = np.zeros((640*3, 640*3, 3), dtype=np.uint8)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // 3
        col = i % 3
        
        y_offset = row * 640
        x_offset = col * 640
        mosaic_img[y_offset:y_offset+640, x_offset:x_offset+640] = img
        
        # 调整标签坐标
        for box in label:
            box[0] += x_offset
            box[1] += y_offset
    
    return mosaic_img, adjusted_labels
```

**效果**：
```
原始：4 个小目标
Mosaic-4：最多 16 个小目标（4 张图的组合）
Mosaic-9：最多 36 个小目标（9 张图的组合）
```

### 其他有效增强
1. **随机缩放**：0.5-2.0 倍，让小目标变大
2. **随机裁剪**：保留多个小目标的组合
3. **Color Jitter**：增加小目标的可见性
4. **CutMix**：合成包含小目标的新样本

---

## 6. 训练策略优化

### 两阶段训练

```python
# 第一阶段：大目标优化
optimizer = AdamW(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(50):
    # 训练，使用标准增强
    train_with_large_objects()

# 第二阶段：小目标优化
optimizer.param_groups[0]['lr'] = 0.0001  # 降低学习率
for epoch in range(50, 100):
    # 训练，使用强增强和 Mosaic
    train_with_small_objects()
```

### 硬样本挖掘

```python
class HardExampleMining:
    def __init__(self, ratio=0.1):
        self.ratio = ratio
    
    def mine_hard_examples(self, losses, batch_size):
        """选择损失最大的样本"""
        num_hard = max(1, int(batch_size * self.ratio))
        hard_indices = torch.topk(losses, num_hard)[1]
        return hard_indices
```

### 学习率调度

```python
# 预热 + 余弦衰减
warmup_epochs = 5
total_epochs = 100

def get_lr(epoch):
    if epoch < warmup_epochs:
        return 0.001 * (epoch + 1) / warmup_epochs
    else:
        return 0.001 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))) / 2
```

---

## 7. 多尺度检测

### 多分辨率训练

```python
def multi_scale_training(model, data_loader, epochs=100):
    """在多个分辨率上轮流训练"""
    scales = [416, 512, 608, 640, 704, 832, 1024]
    
    for epoch in range(epochs):
        scale = scales[epoch % len(scales)]
        model.input_size = scale
        
        # 按这个分辨率训练...
        train_one_epoch(model, data_loader)
```

### Test Time Augmentation (TTA)

在推理时进行多尺度测试，提高检测率。

```python
def tta_detect(model, image, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
    """测试时多尺度增强"""
    all_detections = []
    
    for scale in scales:
        # 缩放图像
        h, w = image.shape[:2]
        scaled_img = cv2.resize(image, (int(w*scale), int(h*scale)))
        
        # 推理
        detections = model(scaled_img)
        
        # 缩放回原始大小
        detections[:, :4] /= scale
        all_detections.append(detections)
    
    # 合并所有检测，使用 NMS
    all_detections = np.vstack(all_detections)
    final_detections = nms(all_detections, iou_threshold=0.5)
    
    return final_detections
```

---

## 8. 后处理优化

### 改进的 NMS

```python
def weighted_nms(detections, iou_threshold=0.5, confidence_threshold=0.5):
    """加权 NMS：保留有竞争的高置信度框"""
    detections = detections[detections[:, 4] > confidence_threshold]
    detections = detections[np.argsort(-detections[:, 4])]
    
    keep = []
    while len(detections) > 0:
        keep.append(detections[0])
        
        if len(detections) == 1:
            break
        
        # 计算 IoU
        ious = compute_iou(detections[0], detections[1:])
        
        # 保留 IoU 较小的（不重叠的）
        detections = detections[1:][ious < iou_threshold]
    
    return np.array(keep)
```

### Soft-NMS

不是直接删除重叠框，而是降低其置信度。

```python
def soft_nms(detections, iou_threshold=0.5, sigma=0.5):
    """Soft-NMS：保留所有框，但降低重叠框的得分"""
    detections = detections[np.argsort(-detections[:, 4])]
    
    for i in range(len(detections)):
        for j in range(i+1, len(detections)):
            iou = compute_iou(detections[i], detections[j])
            
            if iou > iou_threshold:
                # 高斯加权：IoU 越大，惩罚越大
                weight = np.exp(-(iou**2) / sigma)
                detections[j, 4] *= weight
    
    # 按置信度排序后处理
    return detections[detections[:, 4] > confidence_threshold]
```

---

## 9. 完整的优化方案

### 综合方案
```python
class SmallObjectDetectionOptimizer:
    def __init__(self):
        self.config = {
            'input_size': 1280,          # 高分辨率
            'use_mosaic': True,          # Mosaic 增强
            'use_focal_loss': True,      # Focal Loss
            'backbone': 'efficientnet',  # 高效骨干
            'neck': 'pafpn',            # 高级颈部
            'num_scales': 5,            # 5 个特征尺度
            'use_tta': True,            # 测试时增强
            'nms_method': 'soft_nms',   # 软 NMS
        }
    
    def train(self, model, train_loader):
        # 使用优化的超参数训练
        pass
    
    def infer(self, model, image):
        # 使用 TTA 推理
        pass
```

---

## 10. 性能对比

| 方法 | mAP@small | 计算量 | 实时性 |
|------|----------|--------|--------|
| 基准（640） | 60% | 1× | ✓✓✓ |
| +高分辨率（1280） | 68% | 3× | ✓ |
| +Mosaic | 72% | 3× | ✓ |
| +多尺度 FPN | 75% | 3.5× | ✓ |
| +TTA | 78% | 10× | ✗ |

---

## 实战检查清单

- [ ] 输入分辨率提到 1024+
- [ ] Mosaic 增强启用
- [ ] FPN 添加超小尺度特征
- [ ] 锚点针对数据集优化
- [ ] 使用 Focal Loss
- [ ] 两阶段训练策略
- [ ] Test Time Augmentation
- [ ] Soft-NMS 后处理

---

## 参考资源

- 论文：*Small-Object Detection in Unmanned Aerial Images using End-to-End Convolutional Neural Networks*
- YOLOv5 官方文档：Small object detection tips

---

## 关键链接

- [[./YOLO完整深度解析|YOLO 完整解析]]
- [[./Data_Augmentation_Strategies|数据增强策略]]
- [[./YOLO_Loss_Functions|损失函数设计]]

---

**学习建议**：小目标检测需要综合多个技术。从高分辨率 + Mosaic 开始，逐步加入其他优化。
