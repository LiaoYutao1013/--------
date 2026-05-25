# 🎭 非线性滤波与形态学操作：实战项目

来源笔记：`01_Foundations (底层基石)/Digital_Image_Processing (OpenCV基础、滤波、颜色空间)/Image_Filtering_&_Enhancement/非线性滤波器.md`

## 项目目标

构建一个 16x16 合成图像实验台，验证《🎭 非线性滤波与形态学操作》中的像素处理思路。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 概述
- 1. 中值滤波（Median Filter）
- 2. 双边滤波（Bilateral Filter）
- 3. 形态学操作（Morphological Operations）
- 4. 滤波对比与选择
- 5. 实际应用案例
- 6. 性能优化建议
- 自测练习

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
