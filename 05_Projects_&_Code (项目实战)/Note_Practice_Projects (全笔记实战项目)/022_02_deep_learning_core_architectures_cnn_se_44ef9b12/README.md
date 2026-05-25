# 🏗️ ResNet 残差学习完全指南：实战项目

来源笔记：`02_Deep_Learning_Core (神经网络核心)/Architectures (模型架构)/CNN_Series (ResNet, EfficientNet, MobileNet)/ResNet.md`

## 项目目标

搭建轻量结构分析器，统计《🏗️ ResNet 残差学习完全指南》相关模块的形状和参数量。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 概述
- 1. 问题背景
- 2. 残差学习的核心思想
- 3. ResNet 架构
- 4. PyTorch 完整实现
- 5. 性能分析
- 6. 改进变体
- 7. 迁移学习应用

## 运行

```bash
python main.py
```

运行后会生成：

- `artifacts/result.json`

## 可扩展方向

- 把合成数据替换成真实数据。
- 把纯 Python 实现替换成 OpenCV、PyTorch、ONNX Runtime 或真实项目代码。
- 增加单元测试和指标阈值，让实验可以持续复现。
