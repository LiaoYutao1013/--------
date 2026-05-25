# YOLO 算法原理全深度解析：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/Object_Detection/YOLO完整深度解析.md`

## 项目目标

构建合成检测框流水线，验证《YOLO 算法原理全深度解析》涉及的框、损失或后处理逻辑。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 📌 概述
- 🔴 YOLO v1 - 开创性工作
- 🟠 YOLO v2/v3 - 持续改进
- 🟡 YOLO v4/v5 - 工程优化
- 🟢 YOLO v6/v7/v8+ - 最新进展
- 📊 YOLO 系列对比
- 🔗 关键技术深入
- 📈 损失函数详解

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
