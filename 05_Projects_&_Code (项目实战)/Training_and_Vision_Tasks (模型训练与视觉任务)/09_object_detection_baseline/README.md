# 目标检测框回归 Baseline

## 任务目标

训练一个合成检测框回归器，并计算 IoU。

## 覆盖内容

- 训练方式：`detection`
- 视觉任务：`object_detection`
- 数据来源：脚本内合成数据，可独立运行
- 运行产物：`artifacts/result.json`

## 运行

```bash
python main.py
```

## 建议迁移方向

- 替换成 YOLO 格式标签
- 加入 anchor/anchor-free 对比
- 使用 mAP 代替单一 IoU
