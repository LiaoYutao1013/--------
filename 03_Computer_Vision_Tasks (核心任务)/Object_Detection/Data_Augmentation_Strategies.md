# 数据增强策略完全指南

## 概述

数据增强是提升模型性能的关键技术。它通过人工生成新的训练样本，增加数据的多样性，改进模型的泛化能力。

对于 YOLO 检测，好的增强策略能显著提升小目标检测和复杂场景的表现。

## 基础增强方法

### 1. 几何变换

#### 旋转（Rotation）
```python
import cv2
import numpy as np

def rotate_image(img, angle):
    """旋转图像"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))
```

**参数**：
- 旋转角度：±5° 到 ±30°（推荐 ±15°）
- 对旋转不敏感的场景（如竖拍）可增大范围

#### 缩放（Scaling）
```python
def scale_image(img, scale_factor):
    """缩放图像"""
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    return cv2.resize(img, (new_w, new_h))
```

**参数**：
- 缩放范围：0.8 到 1.2（推荐）
- 帮助模型学习多尺度目标

#### 裁剪（Cropping）
```python
def random_crop(img, crop_size):
    """随机裁剪"""
    h, w = img.shape[:2]
    y = np.random.randint(0, h - crop_size)
    x = np.random.randint(0, w - crop_size)
    return img[y:y+crop_size, x:x+crop_size]
```

**参数**：
- 裁剪比例：0.7 到 1.0
- 可能丢失目标，谨慎使用

#### 翻转（Flipping）
```python
def flip_image(img, flip_h=True, flip_v=False):
    """翻转图像"""
    if flip_h:
        img = cv2.flip(img, 1)  # 水平翻转
    if flip_v:
        img = cv2.flip(img, 0)  # 垂直翻转
    return img
```

**参数**：
- 水平翻转：通常总是使用（概率 0.5）
- 垂直翻转：取决于场景（如行人检测不推荐）

#### 透视变换（Perspective）
```python
def perspective_transform(img, strength=0.1):
    """透视变换模拟不同视角"""
    h, w = img.shape[:2]
    
    # 随机四个角的偏移
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    offset = int(strength * min(h, w))
    pts2 = pts1 + np.random.uniform(-offset, offset, size=pts1.shape)
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h))
```

**应用**：自动驾驶（不同视角的车辆）

### 2. 颜色增强

#### 亮度调整
```python
def adjust_brightness(img, brightness_factor):
    """调整亮度"""
    return cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
```

**参数**：
- 范围：0.7 到 1.3
- 模拟不同光照条件

#### 对比度调整
```python
def adjust_contrast(img, contrast_factor):
    """调整对比度"""
    mean = img.mean()
    return np.clip((img - mean) * contrast_factor + mean, 0, 255)
```

**参数**：
- 范围：0.8 到 1.2

#### 色调调整（HSV 空间）
```python
def adjust_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    """在 HSV 空间调整颜色"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    H = (H + 180 * np.random.uniform(-h_gain, h_gain)) % 180
    S = np.clip(S * np.random.uniform(1 - s_gain, 1 + s_gain), 0, 255)
    V = np.clip(V * np.random.uniform(1 - v_gain, 1 + v_gain), 0, 255)
    
    hsv[..., 0], hsv[..., 1], hsv[..., 2] = H, S, V
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
```

**参数**：
- H 范围：±0.5
- S 范围：×0.7 到 ×1.3
- V 范围：×0.7 到 ×1.3

#### 高斯模糊
```python
def gaussian_blur(img, kernel_size=5):
    """高斯模糊"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```

**应用**：模拟运动模糊、焦外

---

## 高级增强方法

### Cutout
移除图像的随机部分，增强模型的鲁棒性。

```python
def cutout(img, num_holes=1, hole_size=32):
    """Cutout 增强"""
    img = img.copy()
    h, w = img.shape[:2]
    
    for _ in range(num_holes):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        
        y1 = max(0, y - hole_size // 2)
        y2 = min(h, y + hole_size // 2)
        x1 = max(0, x - hole_size // 2)
        x2 = min(w, x + hole_size // 2)
        
        img[y1:y2, x1:x2] = 0  # 填充黑色或均值
    
    return img
```

**参数**：
- 洞的数量：1-3
- 洞的大小：32×32 到 64×64

**优势**：
- 增强对遮挡的鲁棒性
- 参数极少

### Mixup
混合两张图像和标签。

$$x' = \lambda x_i + (1-\lambda) x_j$$
$$y' = \lambda y_i + (1-\lambda) y_j$$

其中 $\lambda \sim \text{Beta}(\alpha, \alpha)$，通常 $\alpha = 1.2$

```python
def mixup(img1, label1, img2, label2, alpha=1.2):
    """Mixup 增强"""
    lam = np.random.beta(alpha, alpha)
    img = (lam * img1 + (1 - lam) * img2).astype(np.uint8)
    # 标签混合处理
    return img
```

**优势**：
- 改进模型的线性性
- 提高泛化能力

**注意**：
- 在目标检测中，标签混合需要特殊处理
- 通常只混合背景较多的样本

### CutMix
剪切一张图像的随机矩形区域，粘贴到另一张图像上。

```python
def cutmix(img1, label1, img2, label2, alpha=1.0):
    """CutMix 增强"""
    lam = np.random.beta(alpha, alpha)
    
    h, w = img1.shape[:2]
    cut_h = int(h * np.sqrt(1 - lam))
    cut_w = int(w * np.sqrt(1 - lam))
    
    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)
    
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    
    img = img1.copy()
    img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    
    return img
```

**优势**：
- 比 Mixup 更强的约束
- 保留更多的原始信息

### 使用库的简便方式
```python
from albumentations import Compose, HorizontalFlip, Rotate, ColorJitter

transform = Compose([
    HorizontalFlip(p=0.5),
    Rotate(limit=10, p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
], bbox_params=A.BboxParams(format='pascal_voc'))
```

---

## YOLO 专用增强

### Mosaic 增强 ⭐ 重要

将 4 张图像（或 9 张）拼接成一张，极大增加上下文信息。

```python
def mosaic_augment(images, labels, num_patches=4):
    """Mosaic 增强：4 张图合并成 1 张"""
    h, w = images[0].shape[:2]
    
    if num_patches == 4:
        # 2×2 拼接
        img_out = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        positions = [(0, 0), (w, 0), (0, h), (w, h)]
        
        for i, (x, y) in enumerate(positions):
            img_out[y:y+h, x:x+w] = images[i]
        
        # 标签也需要相应调整
        return img_out
    
    elif num_patches == 9:
        # 3×3 拼接
        img_out = np.zeros((h*3, w*3, 3), dtype=np.uint8)
        positions = [(i*w, j*h) for i in range(3) for j in range(3)]
        for i, (x, y) in enumerate(positions):
            img_out[y:y+h, x:x+w] = images[i]
        return img_out
```

**优势**：
- 大幅增加批量大小的上下文
- YOLOv4+ 默认使用
- 特别有效提升小目标检测

**实现**：
```
+-------+-------+
| img1  | img2  |
+-------+-------+
| img3  | img4  |
+-------+-------+
       ↓
   (拼接后有 4 倍的背景上下文)
```

### Random Affine
随机仿射变换，包括旋转、缩放、平移。

```python
def random_affine(img, targets, 
                  degrees=10, translate=0.1, 
                  scale=0.1, shear=2):
    """随机仿射变换"""
    h, w = img.shape[:2]
    
    # 旋转
    angle = np.random.uniform(-degrees, degrees)
    # 缩放
    scale_factor = np.random.uniform(1-scale, 1+scale)
    # 平移
    tx = np.random.uniform(-translate * w, translate * w)
    ty = np.random.uniform(-translate * h, translate * h)
    
    # 组合变换矩阵...
    
    return img, targets
```

---

## 数据增强策略总结

### 推荐的增强组合

```python
# 基础版本
basic_aug = [
    'HorizontalFlip(p=0.5)',
    'RandomRotate90(p=0.1)',
    'GaussNoise(p=0.1)',
    'Blur(blur_limit=3, p=0.1)',
]

# 中等强度
medium_aug = basic_aug + [
    'ColorJitter(brightness=0.2, contrast=0.2, p=0.3)',
    'RandomBrightnessContrast(p=0.2)',
    'Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=0.3)',
]

# 强增强（YOLO 推荐）
strong_aug = medium_aug + [
    'Mosaic(p=0.5)',
    'Mixup(p=0.1)',
    'CutMix(p=0.1)',
    'RandomAffine()',
]
```

### 按任务选择增强

| 任务 | 推荐增强 |
|------|---------|
| 分类 | 旋转、颜色、Mixup |
| 检测 | Mosaic、随机仿射、颜色 |
| 分割 | 几何变换、Cutout |
| 小目标 | 缩放、Mosaic、颜色 |

---

## 自动增强（AutoAugment）

自动搜索最优的增强策略。

```python
from autoaugment import CIFAR10Policy
import albumentations as A

# 使用预训练的策略
transform = A.Compose([
    A.AutoAugment(),
])
```

**优势**：
- 无需手工设计
- 通常能找到更好的组合

**缺点**：
- 计算成本高
- 需要大量数据进行搜索

---

## 增强对性能的影响

```
模型精度

     无增强
        ↓
    +基础增强 ↑ +5-10%
        ↓
    +颜色增强 ↑ +2-5%
        ↓
    +高级增强 ↑ +3-8%
        ↓
    +AutoAugment ↑ +1-3%
```

---

## 常见错误

### ❌ 错误 1：过度增强
- 增强强度过大导致图像失真
- 模型学习到不切实际的特征

### ❌ 错误 2：增强不一致
- 训练时增强，测试不增强导致分布差异
- 应该让验证集保持一致性

### ❌ 错误 3：忽视增强的计算成本
- 过于复杂的增强导致训练变慢
- 需要在性能和速度间平衡

---

## PyTorch 中的最佳实践

```python
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            # 增强
            augmented = self.transform(image=img, bboxes=label['bboxes'])
            img = augmented['image']
            label['bboxes'] = augmented['bboxes']
        
        return img, label

# 定义增强
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.ColorJitter(p=0.3),
    A.Mosaic(p=0.5),
    A.Normalize(),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='coco'))

# 创建数据加载器
train_dataset = CustomDataset(images, labels, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

---

## 参考资源

- **论文**：
  - Cutout: https://arxiv.org/abs/1708.04552
  - Mixup: https://arxiv.org/abs/1710.09412
  - CutMix: https://arxiv.org/abs/1905.04412
  - Mosaic: YOLOv4 论文

- **库**：
  - Albumentations：https://github.com/albumentations-team/albumentations
  - imgaug：https://github.com/aleju/imgaug

---

## 关键链接

- [[./YOLO完整深度解析|YOLO 完整解析]]
- [[../../02_Deep_Learning_Core (神经网络核心)/Training_Techniques|训练技巧]]

---

**学习建议**：从简单的增强开始，逐步增加复杂性。通过对比实验找到对你的数据集最有效的增强策略。

