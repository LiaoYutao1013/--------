# 冻结特征的迁移学习

## 任务目标

模拟固定 backbone，只训练分类头的迁移学习流程。

## 覆盖内容

- 训练方式：`frozen_transfer`
- 视觉任务：`classification`
- 数据来源：脚本内合成数据，可独立运行
- 运行产物：`artifacts/result.json`

## 运行

```bash
python main.py
```

## 建议迁移方向

- 替换为 PyTorch Dataset/DataLoader
- 把逻辑回归替换成 CNN/ResNet
- 加入混淆矩阵和错误样本可视化
