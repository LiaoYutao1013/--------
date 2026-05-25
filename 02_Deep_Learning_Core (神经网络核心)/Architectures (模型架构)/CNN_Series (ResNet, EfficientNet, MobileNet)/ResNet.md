# 🏗️ ResNet 残差学习完全指南

## 概述

ResNet（Residual Network）通过**跳跃连接（Skip Connection）**解决深层网络的梯度消失问题，首次成功训练超过 100 层的深度神经网络。

**学习时间**：3-4 小时  
**先修课程**：[[卷积神经网络CNN基础|经典 CNN 架构]]、[[../../../../Math_for_CV (线性代数、概率论、最优化)/最优化|最优化]]

---

## 1. 问题背景

### 1.1 深度网络的困境

**观察**：简单堆叠卷积层并不能无限提高网络性能。

```
网络深度 vs 准确率
┌─────────────────────┐
│                  ╱╲ │  更深 ≠ 更好
│              ╱╲╱  │  (degradation problem)
│          ╱╲╱      │  
│      ╱╲╱          │
│  ╱╲╱              │
└─────────────────────┘
  网络深度
```

**两个关键问题**：
1. **梯度消失**：深层网络梯度传播困难
2. **特征退化**：随意加深会降低性能

### 1.2 为什么更深会更差？

$$L = f_n(f_{n-1}(...f_1(x)))$$

反向传播：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f_n} \cdot \frac{\partial f_n}{\partial f_{n-1}} \cdot ... \cdot \frac{\partial f_1}{\partial x}$$

如果 $\frac{\partial f_i}{\partial f_{i-1}} < 1$，则梯度呈指数衰减。

---

## 2. 残差学习的核心思想

### 2.1 恒等映射（Identity Mapping）

**关键洞察**：学习残差比直接学习映射更容易。

**传统做法**：
$$y = H(x)$$

**ResNet 做法**：
$$y = H(x) + x$$

其中 $H(x)$ 学习**残差**（residual），即 $H(x) = y - x$。

**直观理解**：
- 如果 $y \approx x$，则 $H(x) \approx 0$（容易学习）
- 梯度反向传播可以直接绕过多个层

### 2.2 数学分析

假设 $y = x + H(x)$，反向传播时：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \left(1 + \frac{\partial H}{\partial x}\right)$$

关键：**梯度总有 $+1$ 项**，保证了梯度流动。

即使 $\frac{\partial H}{\partial x}$ 很小，梯度也不会完全消失。

---

## 3. ResNet 架构

### 3.1 残差块（Residual Block）

#### 基础残差块（Basic Block）

```
┌─ x ─┐
│     │
▼     │
Conv  │ (out_channel = in_channel)
ReLU  │
Conv  │
│     │
└─+─┬─┘ (相加)
  │ │
  ▼ ▼
  H(x) + x
    │
    ▼
  ReLU
```

数学表达：
$$y = \text{ReLU}(x + F(x))$$

其中 $F(x)$ 是两层卷积的残差函数。

#### 瓶颈残差块（Bottleneck Block）

用于更深的网络（50 层+），减少计算量：

```
┌─ x ──────┐
│          │
▼          │
Conv 1×1   │ (降维，减少通道)
ReLU       │
Conv 3×3   │
ReLU       │
Conv 1×1   │ (升维，恢复通道)
│          │
└─+─┬──────┘ (相加)
  │ │
  ▼ ▼
  y + x
    │
    ▼
  ReLU
```

### 3.2 维度匹配

当输入和输出通道不同时，需要**投影**：

$$y = F(x) + W_s x$$

其中 $W_s$ 是 1×1 卷积用于调整维度。

```python
# 两种做法

# 方法 1：1×1 卷积投影
shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

# 方法 2：填充零
# 如果 out_channels > in_channels，则零填充
```

### 3.3 网络结构

ResNet 家族：ResNet-18, 34, 50, 101, 152

**ResNet-50 结构**：

```
输入(224×224×3)
  │
  ▼
Conv 7×7, stride=2 + ReLU + MaxPool
  │
  ▼
┌─── Bottleneck Block × 3 (256 channels) ─────┐
│                                              │
└──────────────────────────────────────────────┘
  │
  ▼
┌─── Bottleneck Block × 4 (512 channels, stride=2) ─┐
│                                                    │
└────────────────────────────────────────────────────┘
  │
  ▼
┌─── Bottleneck Block × 6 (1024 channels, stride=2) ┐
│                                                    │
└────────────────────────────────────────────────────┘
  │
  ▼
┌─── Bottleneck Block × 3 (2048 channels, stride=2) ┐
│                                                    │
└────────────────────────────────────────────────────┘
  │
  ▼
Average Pooling + FC(1000)
  │
  ▼
Softmax
```

**参数统计**：

| ResNet | 层数 | 基础块 | 参数 | Top-5 错误 |
|--------|------|--------|------|-----------|
| ResNet-18 | 18 | Basic | 11.7M | 10.97% |
| ResNet-34 | 34 | Basic | 21.8M | 7.77% |
| ResNet-50 | 50 | Bottleneck | 25.5M | 5.25% |
| ResNet-101 | 101 | Bottleneck | 44.5M | 4.87% |
| ResNet-152 | 152 | Bottleneck | 60.2M | 4.51% |

---

## 4. PyTorch 完整实现

### 4.1 基础残差块

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1  # 输出通道 / 输入通道
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 映射层（用于维度调整）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        identity = self.shortcut(identity)
        out = out + identity
        out = self.relu(out)
        
        return out
```

### 4.2 瓶颈块

```python
class Bottleneck(nn.Module):
    expansion = 4  # 输出通道 = 输入通道 × 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        
        # 1×1 卷积降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3×3 卷积主体
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1×1 卷积升维
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 映射层
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        identity = x
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 残差连接
        identity = self.shortcut(identity)
        out = out + identity
        out = self.relu(out)
        
        return out
```

### 4.3 完整 ResNet 网络

```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        
        # 第一个块可能改变步长
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        
        # 后续块保持步长为 1
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 分类器
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# 构造不同大小的 ResNet
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
```

---

## 5. 性能分析

### 5.1 残差连接的作用

```python
def analyze_residual_connection():
    """分析残差连接对梯度的影响"""
    
    # 不使用残差连接的梯度
    # ∂L/∂x = ∂L/∂y₅ · ∂y₅/∂y₄ · ... · ∂y₁/∂x
    # 如果每项都 < 1，梯度呈指数衰减
    
    # 使用残差连接的梯度
    # ∂L/∂x = ∂L/∂y (1 + ∂F/∂x)
    # 由于 +1 项的存在，即使 ∂F/∂x 很小，梯度也能传播
    
    print("梯度流动对比")
    print("不使用残差: 0.9^50 =", 0.9**50)  # 极小
    print("使用残差: min(1 + ∂F/∂x) > 0.5 =", 0.5**50, "更稳定")
```

### 5.2 训练曲线对比

```python
def plot_training_curves():
    """ResNet 与 VGG 的训练曲线对比"""
    
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 模拟训练曲线
    epochs = range(1, 101)
    
    # VGG-152 （深层网络无残差连接）
    vgg_train_loss = [1.0 - 0.008*e for e in epochs]
    vgg_val_acc = [0.3 + 0.006*e for e in epochs]
    
    # ResNet-152 （深层网络有残差连接）
    resnet_train_loss = [1.0 - 0.015*e for e in epochs]
    resnet_val_acc = [0.3 + 0.012*e for e in epochs]
    
    ax1.plot(epochs, vgg_train_loss, label='VGG-152', marker='o')
    ax1.plot(epochs, resnet_train_loss, label='ResNet-152', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.legend()
    ax1.set_title('Training Loss Comparison')
    
    ax2.plot(epochs, vgg_val_acc, label='VGG-152', marker='o')
    ax2.plot(epochs, resnet_val_acc, label='ResNet-152', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.legend()
    ax2.set_title('Validation Accuracy Comparison')
    
    plt.tight_layout()
    plt.show()
```

---

## 6. 改进变体

### 6.1 Pre-Activation ResNet

**改进**：将 Batch Norm 和 ReLU 放在卷积前：

```
原始顺序：Conv → BN → ReLU → Conv → BN
改进顺序：BN → ReLU → Conv → BN → ReLU → Conv
```

```python
class PreActBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBlock, self).__init__()
        
        # Pre-activation
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, padding=1)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=1, padding=1)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 
                                     kernel_size=1, stride=stride)
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = out + self.shortcut(x)
        return out
```

### 6.2 Wide ResNet

使用更宽的通道而不是堆叠更多层，计算效率更高。

---

## 7. 迁移学习应用

```python
def use_pretrained_resnet():
    """使用预训练 ResNet 进行迁移学习"""
    
    import torchvision.models as models
    
    # 加载预训练权重
    model = models.resnet50(pretrained=True)
    
    # 冻结所有卷积层
    for param in model.parameters():
        param.requires_grad = False
    
    # 修改分类头
    num_classes = 10
    model.fc = nn.Linear(2048, num_classes)
    
    # 只训练新层
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01)
    
    return model
```

---

## 8. 自测练习

- [x] 从零实现 ResNet-18 的基础块
- [x] 详细推导残差连接对梯度的影响
- [x] 对比有/无残差连接的训练曲线
- [ ] 在自定义数据集上训练 ResNet
- [ ] 实现 Pre-Activation ResNet

---

## 参考资源

- He et al. (2015), Deep Residual Learning for Image Recognition
- He et al. (2016), Identity Mappings in Deep Residual Networks
- 相关主题：[[卷积神经网络CNN基础|经典 CNN 架构]]、[[../../../../02_Deep_Learning_Core (神经网络核心)/Training_Techniques (数据增强、优化器、正则化、损失函数)|训练技巧]]

---

**最后更新**：2026年4月  
**难度等级**：⭐⭐⭐⭐⭐（非常高）  
**实用性**：⭐⭐⭐⭐⭐（极高）
