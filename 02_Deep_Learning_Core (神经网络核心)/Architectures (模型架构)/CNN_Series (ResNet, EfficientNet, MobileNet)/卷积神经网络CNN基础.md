# 🧠 经典 CNN 架构完全指南

## 概述

卷积神经网络（CNN）是深度学习的基础架构。本文详解 CNN 的演变过程：LeNet → AlexNet → VGGNet → ResNet。

**学习时间**：3-4 小时  
**先修课程**：[[../../../../Math_for_CV (线性代数、概率论、最优化)/线性代数|线性代数]]、[[../../../../Math_for_CV (线性代数、概率论、最优化)/最优化|最优化]]

---

## 1. CNN 基础回顾

### 1.1 卷积层

$$y = \sigma(W * x + b)$$

其中 $W$ 是卷积核，$*$ 是卷积操作，$\sigma$ 是激活函数。

**卷积的特性**：
- 局部感受野（Local Receptive Field）
- 权值共享（Parameter Sharing）
- 空间不变性（Spatial Invariance）

### 1.2 池化层

**最大池化**：$y_{ij} = \max\{x_{ij+k} : k \in \text{pool region}\}$

**平均池化**：$y_{ij} = \text{mean}\{x_{ij+k} : k \in \text{pool region}\}$

### 1.3 激活函数

| 激活函数 | 公式 | 特点 |
|---------|------|------|
| **ReLU** | $\max(0, x)$ | 快速收敛，易稀疏 |
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | 平滑，梯度消失 |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | 平滑，居中 |
| **Leaky ReLU** | $\max(\alpha x, x)$ | 改进 ReLU |

---

## 2. LeNet-5 (1998)

### 2.1 网络结构

```
输入(32×32)
    ↓
Conv(6, 5×5) + ReLU
    ↓
Pool(2×2, avg)
    ↓
Conv(16, 5×5) + ReLU
    ↓
Pool(2×2, avg)
    ↓
Conv(120, 5×5) + ReLU
    ↓
FC(84) + ReLU
    ↓
FC(10) + Softmax
    ↓
输出(10类)
```

### 2.2 PyTorch 实现

```python
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, padding=0)
        
        # 池化层
        self.pool = nn.AvgPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Input: (batch_size, 1, 32, 32)
        x = self.relu(self.conv1(x))  # (batch_size, 6, 28, 28)
        x = self.pool(x)              # (batch_size, 6, 14, 14)
        
        x = self.relu(self.conv2(x))  # (batch_size, 16, 10, 10)
        x = self.pool(x)              # (batch_size, 16, 5, 5)
        
        x = self.relu(self.conv3(x))  # (batch_size, 120, 1, 1)
        
        x = x.view(x.size(0), -1)     # 扁平化
        
        x = self.relu(self.fc1(x))    # (batch_size, 84)
        x = self.fc2(x)               # (batch_size, 10)
        
        return x

# 测试
model = LeNet5()
x = torch.randn(1, 1, 32, 32)
output = model(x)
print(output.shape)  # torch.Size([1, 10])
```

### 2.3 特点与贡献

- ✅ 首个成功应用于手写数字识别（MNIST）
- ✅ 引入卷积层和池化层概念
- ✅ 结构简单，参数少
- ❌ 仅 60K 参数，表达能力有限

---

## 3. AlexNet (2012)

### 3.1 网络结构

```
输入(224×224×3)
    ↓
Conv(96, 11×11, stride=4) + ReLU
    ↓
MaxPool(3×3, stride=2)
    ↓
Conv(256, 5×5, padding=2) + ReLU
    ↓
MaxPool(3×3, stride=2)
    ↓
Conv(384, 3×3, padding=1) + ReLU
    ↓
Conv(384, 3×3, padding=1) + ReLU
    ↓
Conv(256, 3×3, padding=1) + ReLU
    ↓
MaxPool(3×3, stride=2)
    ↓
FC(4096) + ReLU + Dropout(0.5)
    ↓
FC(4096) + ReLU + Dropout(0.5)
    ↓
FC(1000) + Softmax
    ↓
输出(1000类)
```

**参数量**：60M

### 3.2 PyTorch 实现

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 统计参数量
model = AlexNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.1f}M")
```

### 3.3 创新点

| 创新 | 作用 |
|------|------|
| **ReLU 激活** | 解决梯度消失，加快收敛 |
| **GPU 并行** | 首个 GPU 实现，2 块 GTX 580 |
| **Data Augmentation** | 数据增强防止过拟合 |
| **Dropout** | 正则化，减少过拟合 |
| **大规模数据集** | ImageNet 1.2M 图像，1000 类 |

**性能**：在 ImageNet 2012 上达到 15.3% top-5 错误率（相比传统方法 ~26%）

---

## 4. VGGNet (2014)

### 4.1 核心思想

**用小卷积核（3×3）的堆叠替代大卷积核**：
- 3 个 3×3 卷积 = 1 个 7×7 卷积（感受野相同）
- 但参数更少，非线性更强

### 4.2 网络结构（VGG-16）

```
输入(224×224×3)
    ↓
Block1: Conv(64, 3×3) × 2 + MaxPool
    ↓
Block2: Conv(128, 3×3) × 2 + MaxPool
    ↓
Block3: Conv(256, 3×3) × 3 + MaxPool
    ↓
Block4: Conv(512, 3×3) × 3 + MaxPool
    ↓
Block5: Conv(512, 3×3) × 3 + MaxPool
    ↓
FC(4096) + ReLU + Dropout
    ↓
FC(4096) + ReLU + Dropout
    ↓
FC(1000) + Softmax
    ↓
输出(1000类)
```

**参数量**：138M

### 4.3 PyTorch 实现

```python
class VGGNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(VGGNet, self).__init__()
        
        self.features = self._make_layers([
            64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512, 'M',
            512, 512, 512, 'M',
        ])
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
        if init_weights:
            self._initialize_weights()
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# VGG 家族
vgg_configs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
```

### 4.4 特点

| 特点 | 说明 |
|------|------|
| **简洁** | 所有层都是 3×3 卷积或 2×2 池化 |
| **深度** | VGG-19 有 19 层，是当时最深的 |
| **表达力强** | 参数多（138M），但组织有序 |
| **易实现** | 结构规则，易于复现 |
| **迁移学习** | 预训练模型在许多任务中有效 |

**性能**：ImageNet top-5 错误率 7.3%

---

## 5. 架构演变对比

### 5.1 参数与性能对比

| 架构 | 年份 | 层数 | 参数 | 计算量 | ImageNet Top-5 |
|------|------|------|------|--------|----------------|
| **LeNet-5** | 1998 | 7 | 60K | 370K | - |
| **AlexNet** | 2012 | 8 | 60M | 1.4G | 15.3% |
| **VGGNet-16** | 2014 | 16 | 138M | 15.3G | 7.3% |
| **ResNet-50** | 2015 | 50 | 25.5M | 3.86G | 5.5% |
| **DenseNet-121** | 2017 | 121 | 7.98M | 2.87G | 5.2% |

### 5.2 架构发展趋势

```
LeNet (7层, 60K参数)
    ↓ 增加层数和通道数
AlexNet (8层, 60M参数)
    ↓ 堆叠小核卷积
VGGNet (16/19层, 138M参数)
    ↓ 残差连接
ResNet (34-152层, 25.5M参数)
    ↓ 稠密连接
DenseNet (121层, 7.98M参数)
```

**关键转变**：
1. **更深** → 层数从 7 到 152+
2. **更聪明** → 通过跳连、稠密连接保持效率
3. **参数更少** → 从 138M 优化到 7.98M
4. **速度更快** → 从 15.3G FLOPs 优化到 2.87G

---

## 6. 实际应用

### 6.1 迁移学习

```python
import torchvision.models as models
import torch.optim as optim

# 加载预训练模型
model = models.vgg16(pretrained=True)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 修改最后一层用于自定义任务
num_classes = 10
model.classifier[-1] = nn.Linear(4096, num_classes)

# 只训练新层
optimizer = optim.SGD(model.classifier[-1].parameters(), lr=0.01)
```

### 6.2 特征可视化

```python
def visualize_features(model, img):
    """可视化 CNN 特征"""
    
    # 获取中间层输出
    layers = {
        'conv1': model.features[0],
        'conv3': model.features[3],
        'conv5': model.features[5],
    }
    
    activations = {}
    
    def hook(name):
        def forward_hook(model, input, output):
            activations[name] = output.detach()
        return forward_hook
    
    for name, layer in layers.items():
        layer.register_forward_hook(hook(name))
    
    model(img)
    
    # 绘制特征图
    for name, activation in activations.items():
        print(f"{name}: {activation.shape}")

def visualize_filters(model):
    """可视化卷积核"""
    
    first_conv = model.features[0]
    filters = first_conv.weight.data.cpu()
    
    # filters shape: (96, 3, 11, 11)
    # 归一化到 [0, 1]
    filters = (filters - filters.min()) / (filters.max() - filters.min())
    
    return filters
```

### 6.3 性能分析

```python
def measure_performance(model, input_size=(1, 3, 224, 224)):
    """性能分析"""
    import time
    
    x = torch.randn(input_size).cuda()
    model = model.cuda()
    model.eval()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    # 计算 FLOPs（需要 fvcore 库）
    from fvcore.nn import FlopCounterMode
    
    with FlopCounterMode(model, display=True):
        y = model(x)
    
    # 测试推理时间
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            y = model(x)
        avg_time = (time.time() - start) / 100 * 1000  # ms
    
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Average inference time: {avg_time:.2f}ms")
```

---

## 7. 自测练习

- [ ] 从零实现 LeNet-5 并在 MNIST 上训练
- [ ] 实现 AlexNet 的完整训练流程
- [ ] 对比 VGG-11 和 VGG-16 的性能差异
- [ ] 使用预训练 VGG 进行迁移学习
- [ ] 可视化 CNN 的特征图和卷积核

---

## 参考资源

- LeNet: LeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition
- AlexNet: Krizhevsky et al. (2012), ImageNet Classification with Deep Convolutional Neural Networks
- VGGNet: Simonyan & Zisserman (2014), Very Deep Convolutional Networks for Large-Scale Image Recognition
- 相关主题：[[./ResNet|ResNet 残差学习]]、[[../../../../02_Deep_Learning_Core (神经网络核心)/Training_Techniques (数据增强、优化器、正则化、损失函数)|训练技巧]]

---

**最后更新**：2026年4月  
**难度等级**：⭐⭐⭐⭐（高等）  
**实用性**：⭐⭐⭐⭐⭐（极高）
