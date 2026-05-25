# 04 INT8 PTQ 量化误差实验

对应知识点：量化。

这个项目实现一个最小的训练后量化（PTQ）流程：

1. 生成校准样本
2. 统计激活范围
3. 把权重量化为 INT8
4. 对比 FP32 与 INT8 推理输出误差

## 运行

```bash
python main.py
```

可调参数：

```bash
python main.py --calibration-samples 512 --test-samples 128
```

输出文件：`artifacts/quantization_report.json`。
