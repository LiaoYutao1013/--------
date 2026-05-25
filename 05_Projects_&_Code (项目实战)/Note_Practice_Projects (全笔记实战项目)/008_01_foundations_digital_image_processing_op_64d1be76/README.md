# 🔍 边缘检测详解：实战项目

来源笔记：`01_Foundations (底层基石)/Digital_Image_Processing (OpenCV基础、滤波、颜色空间)/Edge_Detection_&_Features/边缘检测详解.md`

## 项目目标

构建合成检测框流水线，验证《🔍 边缘检测详解》涉及的框、损失或后处理逻辑。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 概述
- 1. 边缘检测原理
- 2. Sobel 边缘检测
- 3. Laplacian 边缘检测
- 4. Canny 边缘检测（最重要）
- 5. 其他边缘检测方法
- 6. 实际应用
- 7. 边缘检测对比与选择

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
