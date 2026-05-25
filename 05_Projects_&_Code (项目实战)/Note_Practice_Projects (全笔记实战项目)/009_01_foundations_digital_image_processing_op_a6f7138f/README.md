# 🧠 深度学习笔记：卷积核 (Kernel / Filter)：实战项目

来源笔记：`01_Foundations (底层基石)/Digital_Image_Processing (OpenCV基础、滤波、颜色空间)/Image_Filtering_&_Enhancement/Kernel and Filter.md`

## 项目目标

构建一个 16x16 合成图像实验台，验证《🧠 深度学习笔记：卷积核 (Kernel / Filter)》中的像素处理思路。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 1. 什么是卷积核？
- 2. 核心数学操作：互相关运算
- 3. 卷积核的关键超参数
- 4. 卷积核学到了什么？
- 5. 常见特殊卷积核类型
- 6. 输出尺寸计算公式 📏
- 7. 相关资源与延伸阅读

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
