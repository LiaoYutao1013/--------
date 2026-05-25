# 模型部署总览：实战项目

来源笔记：`05_Projects_&_Code (项目实战)/Deployment (模型部署：TensorRT, ONNX)/模型部署总览.md`

## 项目目标

生成部署清单并执行检查，对应《模型部署总览》的工程化流程。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 部署流水线
- ONNX
- TensorRT
- 量化
- 部署检查清单
- YOLO 部署重点
- 配套实战项目

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
