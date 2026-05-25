# 计算机视觉知识图谱

## 🌐 核心知识体系

### 第一层：数学与信号处理基础
```
数学基础
├── 线性代数
│   ├── 向量与矩阵运算
│   ├── 特征值与特征向量
│   ├── 矩阵分解（SVD、QR）
│   └── 张量运算
├── 概率论与统计
│   ├── 概率分布（高斯、伯努利等）
│   ├── 贝叶斯定理
│   ├── 最大似然估计
│   └── 马尔可夫链
└── 最优化
    ├── 凸优化
    ├── 梯度下降
    ├── 二阶方法
    └── 约束优化
```

### 第二层：图像处理基础
```
数字图像处理
├── 图像表示
│   ├── 颜色空间（RGB、HSV、YUV）
│   ├── 图像编码（JPEG、PNG）
│   └── 采样与量化
├── 图像操作
│   ├── 几何变换（旋转、缩放、仿射）
│   ├── 配准与对齐
│   └── 插值方法
├── 图像滤波
│   ├── 线性滤波
│   ├── 非线性滤波
│   └── 形态学操作
└── 特征提取
    ├── 边缘检测
    ├── 角点检测
    ├── 局部描述符（SIFT、ORB）
    └── 纹理特征
```

### 第三层：机器学习方法
```
机器学习
├── 传统算法
│   ├── 线性模型（SVM、LR）
│   ├── 决策树与集成
│   ├── 聚类（K-Means、GMM）
│   └── 概率图模型
└── 深度学习
    ├── 前向传播
    ├── 反向传播
    ├── 正则化技术
    └── 优化算法
```

### 第四层：深度学习架构
```
深度神经网络
├── CNN 系列
│   ├── 基础：LeNet、AlexNet、VGG
│   ├── 改进：ResNet、EfficientNet、MobileNet
│   ├── 密集：DenseNet、CSPNet
│   └── 轻量级：ShuffleNet、SqueezeNet
├── Transformer 系列
│   ├── 基础：ViT、DeiT
│   ├── 分层：Swin、Hierarchical ViT
│   ├── 自监督：MAE、BEiT
│   └── 混合：CvT、HybridViT
└── 其他架构
    ├── RNN/LSTM
    ├── GRU
    ├── MLP-Mixer
    └── 混合模型
```

### 第五层：CV 核心任务
```
计算机视觉任务
├── 分类（Classification）
│   ├── 单标签分类
│   ├── 多标签分类
│   └── 细粒度分类
├── 检测（Detection）
│   ├── 单阶段：YOLO、SSD
│   ├── 两阶段：R-CNN、Faster RCNN
│   ├── Anchor-free：CenterNet、FCOS
│   └── 追踪：MOT、DeepSORT
├── 分割（Segmentation）
│   ├── 语义分割：FCN、U-Net、DeepLab
│   ├── 实例分割：Mask RCNN、YOLACT
│   └── 全景分割：Panoptic FPN
├── 生成（Generation）
│   ├── GAN 系列
│   ├── Diffusion Models
│   └── Flow Models
└── 3D 视觉
    ├── 几何基础
    ├── 3D 表示
    ├── 神经渲染
    └── 3D 重建
```

### 第六层：工程应用
```
实战应用
├── 数据处理
│   ├── 标注工具
│   ├── 数据增强
│   └── 平衡处理
├── 模型优化
│   ├── 量化
│   ├── 蒸馏
│   └── 剪枝
├── 部署框架
│   ├── ONNX
│   ├── TensorRT
│   ├── TensorFlow Lite
│   └── CoreML
└── 生产系统
    ├── 推理引擎
    ├── 监控告警
    └── 持续学习
```

## 📊 知识点关联图

### 从检测任务出发的知识依赖
```
YOLO 检测
├── 需要理解
│   ├── CNN 特征提取
│   │   ├── 卷积运算
│   │   ├── 激活函数
│   │   ├── 池化层
│   │   └── ResNet/EfficientNet 等骨干网
│   ├── 目标检测框架
│   │   ├── 锚点设置
│   │   ├── NMS 后处理
│   │   ├── 多尺度特征
│   │   └── 损失函数设计
│   ├── 评估指标
│   │   ├── IoU 计算
│   │   ├── mAP 指标
│   │   ├── Precision/Recall
│   │   └── F1-Score
│   └── 训练技巧
│       ├── 数据增强
│       ├── 学习率调度
│       ├── 类别不平衡
│       └── 困难样本挖掘
│
├── 衍生学习
│   ├── 目标跟踪（Kalman Filter）
│   ├── 小目标检测
│   ├── 多尺度融合（FPN）
│   ├── 实例分割（Mask RCNN）
│   └── 3D 目标检测
│
└── 工程实现
    ├── 框架（PyTorch/TF）
    ├── 标注工具
    ├── 模型转换（ONNX）
    └── 部署加速（TensorRT）
```

## 🔗 主题关联

### 按学习难度分类

**初级（1-2 周）**
- [[Math_for_CV|基础数学]]
- [[Image_Representation_&_Color_Spaces|颜色空间与图像表示]]
- [[Pixel_Operations_&_Transformation|基本图像操作]]

**中级（3-8 周）**
- [[Image_Filtering_&_Enhancement|滤波与增强]]
- [[Edge_Detection_&_Features|边缘与特征]]
- [[Architectures/CNN_Series|CNN 架构]]
- [[Machine_Learning_Basic|机器学习基础]]

**高级（9-20 周）**
- [[Object_Detection|目标检测]]
- [[Architectures/Transformer_Series|Transformer 架构]]
- [[Segmentation|图像分割]]
- [[Training_Techniques|高级训练技巧]]

**专家（20+ 周）**
- [[Generative_Models|生成模型]]
- [[3D_Vision|3D 视觉]]
- [[Deployment|模型部署与优化]]

### 按应用领域分类

**医学影像**
- 分割、分类、检测
- 相关：[[Segmentation|语义分割]]、[[Architectures|U-Net]]

**自动驾驶**
- 检测、跟踪、3D 视觉
- 相关：[[Object_Detection|YOLO]]、[[3D_Vision|3D 检测]]、[[Object_Detection/Kalman Filter|目标跟踪]]

**工业检测**
- 缺陷检测、异常检测
- 相关：[[Object_Detection|小目标检测]]、[[Generative_Models|异常检测]]

**人脸识别**
- 人脸检测、识别、属性识别
- 相关：[[Classification|分类]]、[[Segmentation|实例分割]]

**遥感监测**
- 地物分类、变化检测、目标识别
- 相关：[[Segmentation|语义分割]]、[[Object_Detection|目标检测]]

**3D 重建**
- 立体视觉、SLAM、NeRF
- 相关：[[3D_Vision|3D 基础]]、[[3D_Vision/Neural_Rendering_&_Reconstruction|神经渲染]]

## 📈 学习进度跟踪

### 必修基础（必学）
- [x] [[Math_for_CV|数学基础]] - 4 周
- [x] [[Image_Representation_&_Color_Spaces|图像表示]] - 1 周
- [x] [[Image_Filtering_&_Enhancement|滤波处理]] - 2 周
- [x] [[Edge_Detection_&_Features|特征提取]] - 2 周
- [x] [[Architectures/CNN_Series|CNN 基础]] - 3 周
- [x] [[Object_Detection|目标检测]] - 6 周

### 推荐扩展（可选但推荐）
- [ ] [[Architectures/Transformer_Series|Transformer]] - 2 周
- [ ] [[Segmentation|图像分割]] - 3 周
- [ ] [[3D_Vision|3D 视觉]] - 2 周
- [ ] [[Deployment|模型部署]] - 2 周

### 进阶研究（可选）
- [ ] [[Generative_Models|生成模型]] - 3 周
- [ ] [[Training_Techniques|高级训练]] - 2 周

---

> **提示**：点击上方链接快速导航到相应章节。使用 Ctrl+Click 在新标签页打开多个资源进行对比学习。
