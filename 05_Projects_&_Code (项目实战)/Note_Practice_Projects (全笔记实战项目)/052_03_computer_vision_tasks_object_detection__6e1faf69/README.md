# 小目标检测优化完全指南：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/Object_Detection/Small_Object_Detection_Optimization.md`

## 项目目标

构建合成检测框流水线，验证《小目标检测优化完全指南》涉及的框、损失或后处理逻辑。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 问题定义
- 1. 输入分辨率优化
- 2. 特征金字塔改进
- 3. Anchor 优化
- 4. 损失函数优化
- 5. 数据增强策略
- 6. 训练策略优化
- 7. 多尺度检测

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
