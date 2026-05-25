# 3D 表示方法：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/3D_Vision (NeRF, 3DGS, 点云)/3D_Representations/3D表示方法.md`

## 项目目标

用合成 3D 点和相机参数验证《3D 表示方法》相关几何计算。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 常见表示
- 点云
- 网格
- 隐式表示
- NeRF 表示
- 3DGS 表示

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
