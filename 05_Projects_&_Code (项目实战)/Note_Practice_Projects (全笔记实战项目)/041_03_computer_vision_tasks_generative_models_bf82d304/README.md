# 生成模型总览：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/Generative_Models (Diffusion, GAN)/生成模型总览.md`

## 项目目标

用一维信号模拟噪声添加和去噪过程，对应《生成模型总览》的生成建模思想。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 主要路线
- GAN
- Diffusion
- 条件生成
- 与 CV 的关系

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
