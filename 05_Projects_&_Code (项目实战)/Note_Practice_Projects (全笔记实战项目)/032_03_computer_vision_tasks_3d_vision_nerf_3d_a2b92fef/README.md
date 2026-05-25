# 📔 Note: 相机标定与张正友标定法：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/3D_Vision (NeRF, 3DGS, 点云)/3D_Geometry_Foundations/Camera Calibration.md`

## 项目目标

用合成 3D 点和相机参数验证《📔 Note: 相机标定与张正友标定法》相关几何计算。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 1. 为什么要标定？
- 2. 张正友标定法 (Zhang's Method)
- 3. 标定的具体步骤
- 4. 误差来源与注意事项
- 5. 标定结果的应用
- 💡 Obsidian 联动建议

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
