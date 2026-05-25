# Kalman Filter：实战项目

来源笔记：`03_Computer_Vision_Tasks (核心任务)/Object_Detection (检测：YOLO系列、DETR)/Kalman Filter.md`

## 项目目标

构建合成检测框流水线，验证《Kalman Filter》涉及的框、损失或后处理逻辑。

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

- 📖 笔记：卡尔曼滤波 (Kalman Filter) 深度指南
- 2. 数学基石：状态空间表达式
- 3. 卡尔曼滤波的五个黄金公式
- 4. 核心推导：卡尔曼增益 $K$ 是怎么来的？
- 5. 对比：KF vs EKF vs UKF

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
