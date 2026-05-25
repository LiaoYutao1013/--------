# Vision Transformer：实战项目

来源笔记：`02_Deep_Learning_Core (神经网络核心)/Architectures (模型架构)/Transformer_Series (ViT, Swin, MAE)/Vision Transformer.md`

## 项目目标

构建合成检测框流水线，验证《Vision Transformer》涉及的框、损失或后处理逻辑。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 基本流程
- 核心公式
- ViT 与 CNN 的区别
- Swin Transformer
- MAE
- 与检测任务的关系

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
