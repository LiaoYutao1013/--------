# 📔 Note: 坐标系变换 (Coordinate Transformations)：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/3D_Vision (NeRF, 3DGS, 点云)/3D_Geometry_Foundations/Coordinate Transformations.md`

## 项目目标

用合成 3D 点和相机参数验证《📔 Note: 坐标系变换 (Coordinate Transformations)》相关几何计算。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 1. 概述：视觉的“降维打击”
- 2. 变换详解
- 3. 最终组合：相机内参矩阵 (Intrinsic Matrix)
- 4. 全局投影公式
- 💡 避坑指南 (Obsidian Tips)

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
