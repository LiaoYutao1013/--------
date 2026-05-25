# 自监督对比学习预训练

## 任务目标

通过正负样本相似度理解对比学习的训练目标。

## 覆盖内容

- 训练方式：`contrastive`
- 视觉任务：`representation_learning`
- 数据来源：脚本内合成数据，可独立运行
- 运行产物：`artifacts/result.json`

## 运行

```bash
python main.py
```

## 建议迁移方向

- 替换成 SimCLR 风格增强
- 加入温度系数和 batch negatives
- 用线性探针评估特征
