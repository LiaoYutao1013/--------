# 📐 边界框回归损失函数：从 IoU 到 CIoU：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/Object_Detection (检测：YOLO系列、DETR)/YOLO_Loss_Functions.md`

## 项目目标

构建合成检测框流水线，验证《📐 边界框回归损失函数：从 IoU 到 CIoU》涉及的框、损失或后处理逻辑。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 1. 为什么需要 IoU 损失？
- 2. GIoU (Generalized IoU)：解决不相交问题
- 3. DIoU (Distance IoU)：引入中心点距离
- 4. CIoU (Complete IoU)：考虑长宽比
- 5. 性能优劣全维度对比

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
