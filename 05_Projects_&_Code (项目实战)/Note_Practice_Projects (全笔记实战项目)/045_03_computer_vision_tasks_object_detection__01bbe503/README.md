# Small_Object_Detection_Optimization：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/Object_Detection (检测：YOLO系列、DETR)/Small_Object_Detection_Optimization.md`

## 项目目标

构建合成检测框流水线，验证《Small_Object_Detection_Optimization》涉及的框、损失或后处理逻辑。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 1. 策略一：切片辅助推理 (SAHI, Slicing Aided Hyper Inference)
- 2. 策略二：特征融合与多尺度检测 (P2 层的引入)
- 3. 策略三：NWD 损失函数 (Normalized Wasserstein Distance)
- 4. 策略四：特征超分辨率 (Feature Super-Resolution)
- 5. 策略五：背景上下文建模 (Context Modeling)
- 💡 总结与对比表
- 🛠️ 你的 Obsidian 实践建议

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
