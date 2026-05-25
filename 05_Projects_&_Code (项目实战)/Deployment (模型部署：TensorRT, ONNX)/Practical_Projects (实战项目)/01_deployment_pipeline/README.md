# 01 部署流水线模拟器

对应知识点：部署流水线。

这个项目模拟从训练模型到部署包的完整流程：

```text
训练模型 -> 导出 ONNX -> 图优化 -> 推理引擎选择 -> 量化策略 -> 服务化/边缘部署包
```

## 运行

```bash
python main.py
```

可选参数：

```bash
python main.py --target edge --batch-size 1 --input-size 640 640
```

运行后会生成 `artifacts/pipeline_manifest.json`，里面记录每个阶段的输入、输出和关键检查项。
