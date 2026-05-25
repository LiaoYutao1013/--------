# Vision Transformer 在目标检测中的应用：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/Object_Detection/Vision_Transformer_in_Object_Detection.md`

## 项目目标

生成部署清单并执行检查，对应《Vision Transformer 在目标检测中的应用》的工程化流程。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 概述
- 从 ViT 到检测
- ViT 检测的三大范式
- 关键改进方向
- 与 YOLO 的结合
- 实际部署考虑
- 当前研究方向
- 推荐方案

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
