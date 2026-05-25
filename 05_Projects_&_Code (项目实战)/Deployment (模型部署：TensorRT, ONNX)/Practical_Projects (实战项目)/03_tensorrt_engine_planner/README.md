# 03 TensorRT Engine Profile 与命令规划器

对应知识点：TensorRT。

这个项目不要求本机安装 NVIDIA GPU 或 TensorRT。它根据模型结构、目标精度和动态输入尺寸生成一个 TensorRT engine 规划报告，并给出可迁移到真实环境的 `trtexec` 命令。

覆盖点：

- layer fusion 机会识别
- FP16 / INT8 策略选择
- dynamic shape profile 生成
- workspace、batch、shape 的部署记录

## 运行

```bash
python main.py
```

生成文件：

- `artifacts/tensorrt_engine_plan.json`
- `artifacts/trtexec_command.txt`
