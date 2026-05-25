# 图像表示与颜色空间：实战项目

来源笔记：`01_Foundations (底层基石)/Digital_Image_Processing (OpenCV基础、滤波、颜色空间)/Image_Representation_&_Color_Spaces/颜色空间与表示.md`

## 项目目标

构建一个 16x16 合成图像实验台，验证《图像表示与颜色空间》中的像素处理思路。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 概述
- 基本概念
- RGB 颜色空间
- HSV 颜色空间
- 其他颜色空间
- 颜色空间转换矩阵
- Bayer 阵列与图像采集
- 图像尺寸标准

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
