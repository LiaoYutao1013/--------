# 🎯 局部特征提取详解：实战项目

来源笔记：`01_Foundations (底层基石)/Digital_Image_Processing (OpenCV基础、滤波、颜色空间)/Edge_Detection_&_Features/局部特征提取.md`

## 项目目标

构建合成检测框流水线，验证《🎯 局部特征提取详解》涉及的框、损失或后处理逻辑。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 概述
- 1. 特征点的性质
- 2. Harris 角点检测
- 3. SIFT（尺度不变特征变换）
- 4. ORB（Oriented FAST and Rotated BRIEF）
- 5. 特征点对比
- 6. 实际应用
- 7. 性能优化

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
