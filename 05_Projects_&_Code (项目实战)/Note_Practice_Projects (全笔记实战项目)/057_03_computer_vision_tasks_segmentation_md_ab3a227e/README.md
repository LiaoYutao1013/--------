# 图像分割总览：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/Segmentation (分割：语义、实例、全景)/图像分割总览.md`

## 项目目标

生成合成分割 mask，计算《图像分割总览》相关的 IoU 和连通域。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 三类分割
- 典型模型
- 损失函数
- 评价指标
- 与检测的关系

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
