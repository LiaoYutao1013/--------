# CSPNet 架构：跨阶段部分连接：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/Object_Detection/CSPNet_Architecture.md`

## 项目目标

构建合成检测框流水线，验证《CSPNet 架构：跨阶段部分连接》涉及的框、损失或后处理逻辑。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 概述
- 为什么需要 CSPNet？
- CSPNet 核心思想
- YOLOv4 中的 CSPDarknet
- CSPNet 与其他架构的对比
- CSPNet 的关键技术
- 从 CSPDarknet 到 C3 / C2f
- 性能分析

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
