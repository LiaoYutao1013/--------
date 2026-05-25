# YOLO 损失函数深度解析：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/Object_Detection/YOLO_Loss_Functions.md`

## 项目目标

构建合成检测框流水线，验证《YOLO 损失函数深度解析》涉及的框、损失或后处理逻辑。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 概述
- 损失函数的三个部分
- 演变历程
- 现代损失函数
- 类别不平衡的处理
- 位置损失的选择对比
- 完整的 YOLO v4 损失函数
- 调试技巧

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
