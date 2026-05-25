# 模型部署实战项目集

这个目录把《模型部署总览》里的每个知识点拆成一个可运行的小项目。每个项目都可以独立进入目录后运行，不依赖笔记内容。

## 项目列表

| 序号 | 知识点 | 项目 | 运行方式 |
|------|--------|------|----------|
| 01 | 部署流水线 | 部署流水线模拟器 | `python main.py` |
| 02 | ONNX | 导出一致性检查器 | `python main.py` |
| 03 | TensorRT | Engine profile 与 trtexec 命令规划器 | `python main.py` |
| 04 | 量化 | INT8 PTQ 量化误差实验 | `python main.py` |
| 05 | 部署检查清单 | 部署就绪审计器 | `python main.py` |
| 06 | YOLO 部署重点 | Letterbox + NMS 后处理验证器 | `python main.py` |
| 07 | CMake 跨平台部署 | C++/CMake 视觉部署模板 | CMake + `vision_app` |

## 一键运行

```bash
python run_all.py
```

所有脚本都只使用 Python 标准库，适合先理解部署工程要点；后续可以把其中的模拟模型、配置和校验逻辑替换成真实的 PyTorch、ONNX Runtime、TensorRT 或 YOLO 工程。

第 07 个项目是 C++/CMake 模板，用于跨平台部署骨架。
