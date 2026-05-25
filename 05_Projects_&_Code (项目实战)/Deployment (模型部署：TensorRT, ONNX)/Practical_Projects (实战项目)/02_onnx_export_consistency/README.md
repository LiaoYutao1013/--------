# 02 ONNX 导出一致性检查器

对应知识点：ONNX。

这个项目用纯 Python 构造一个小型线性模型，导出成 `model.onnx.json` 这种可读的 ONNX 风格图描述，再用同一份图描述执行推理并和原始模型对比。

重点覆盖：

- opset 版本记录
- 动态 batch 记录
- 算子支持检查
- 导出前后数值一致性检查

## 运行

```bash
python main.py
```

输出目录：`artifacts/`。
