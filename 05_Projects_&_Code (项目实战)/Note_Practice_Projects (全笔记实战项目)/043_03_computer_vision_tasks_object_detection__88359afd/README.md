# 🎨 视觉数据增强：Mixup, CutMix 与 Mosaic：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/Object_Detection (检测：YOLO系列、DETR)/Data_Augmentation_Strategies.md`

## 项目目标

构建合成检测框流水线，验证《🎨 视觉数据增强：Mixup, CutMix 与 Mosaic》涉及的框、损失或后处理逻辑。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 1. Mixup：线性插值的艺术
- 2. CutMix：空间裁剪与替换
- 3. Mosaic：YOLO 的制胜法宝
- 4. 三者的对比与同异

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
