# Note: YOLO 算法原理全深度解析 (Master Deep Dive)：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/Object_Detection (检测：YOLO系列、DETR)/YOLO 算法原理全深度解析 (Master Deep Dive).md`

## 项目目标

构建合成检测框流水线，验证《Note: YOLO 算法原理全深度解析 (Master Deep Dive)》涉及的框、损失或后处理逻辑。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 1. YOLO 的核心哲学：目标检测即回归 (Regression)
- 2. 数学表征：网格与预测值
- 3. 网络架构演进 (The Architecture Evolution)
- 4. 损失函数深度解析 (Loss Function)
- 5. 关键技术细节：为什么 YOLO 这么快？
- 6. 实战学习建议 (Obsidian 工作流)
- 7. 总结：YOLO 的权衡艺术

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
