# Kalman 滤波与目标跟踪：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/Object_Detection/Kalman Filter.md`

## 项目目标

构建合成检测框流水线，验证《Kalman 滤波与目标跟踪》涉及的框、损失或后处理逻辑。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 概述
- 基本原理
- Kalman 滤波算法
- Kalman 增益的直观理解
- 实现示例
- 参数调节
- 与目标检测结合：MOT（多目标跟踪）
- 扩展模型

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
