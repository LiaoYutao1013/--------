# 图像分类总览：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/Classification (分类)/图像分类总览.md`

## 项目目标

生成合成分割 mask，计算《图像分类总览》相关的 IoU 和连通域。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 基本流程
- 分类任务类型
- 常用模型
- 损失函数
- 迁移学习
- 评价指标
- 与检测/分割的关系

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
