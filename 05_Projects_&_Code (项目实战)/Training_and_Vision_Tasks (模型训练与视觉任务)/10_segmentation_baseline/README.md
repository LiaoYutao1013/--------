# 图像分割 Mask Baseline

## 任务目标

生成合成 mask，学习阈值分割、IoU 和 Dice 指标。

## 覆盖内容

- 训练方式：`segmentation`
- 视觉任务：`segmentation`
- 数据来源：脚本内合成数据，可独立运行
- 运行产物：`artifacts/result.json`

## 运行

```bash
python main.py
```

## 建议迁移方向

- 替换成 U-Net 或 DeepLab
- 加入 Dice Loss 和 BCE Loss 对比
- 保存预测 mask 可视化
