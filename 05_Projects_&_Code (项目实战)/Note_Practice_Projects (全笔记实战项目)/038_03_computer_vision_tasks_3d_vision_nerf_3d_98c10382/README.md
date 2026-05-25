# 神经渲染与重建：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/3D_Vision (NeRF, 3DGS, 点云)/Neural_Rendering_&_Reconstruction/神经渲染与重建.md`

## 项目目标

把《神经渲染与重建》转化为可执行任务、里程碑和检查项。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 任务目标
- NeRF
- NeRF 的限制
- 3D Gaussian Splatting
- SLAM 与重建
- 学习路线

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
