# 模型训练与视觉任务

这个模块用于系统练习“怎么训练模型，以及如何完成常见视觉任务”。每个任务都能独立运行，先用纯 Python 合成数据跑通流程，再迁移到 OpenCV/PyTorch/YOLO/U-Net 等真实工程。

## 学习路径

```text
从零训练分类器
  -> mini-batch 与学习率
  -> 冻结特征迁移学习
  -> 微调
  -> 数据增强与超参搜索
  -> 类别不均衡与交叉验证
  -> 检测、分割、自监督
  -> 训练报告与错误分析
```

## 一键运行

```bash
python run_all.py
```

## 任务索引

| ID | 任务 | 训练方式 | 视觉任务 | 入口 |
|----|------|----------|----------|------|
| 01_from_scratch_classifier | 从零训练图像分类器 | `from_scratch` | `classification` | [01_from_scratch_classifier](./01_from_scratch_classifier/README.md) |
| 02_minibatch_training | Mini-batch 训练与学习率实验 | `mini_batch` | `classification` | [02_minibatch_training](./02_minibatch_training/README.md) |
| 03_frozen_feature_transfer | 冻结特征的迁移学习 | `frozen_transfer` | `classification` | [03_frozen_feature_transfer](./03_frozen_feature_transfer/README.md) |
| 04_finetune_last_layers | 解冻后几层进行微调 | `finetune` | `classification` | [04_finetune_last_layers](./04_finetune_last_layers/README.md) |
| 05_augmentation_ablation | 数据增强消融实验 | `augmentation` | `classification` | [05_augmentation_ablation](./05_augmentation_ablation/README.md) |
| 06_hyperparameter_search | 超参数搜索训练任务 | `hyperparameter_search` | `classification` | [06_hyperparameter_search](./06_hyperparameter_search/README.md) |
| 07_class_imbalance_training | 类别不均衡训练 | `class_imbalance` | `classification` | [07_class_imbalance_training](./07_class_imbalance_training/README.md) |
| 08_cross_validation | K 折交叉验证 | `cross_validation` | `classification` | [08_cross_validation](./08_cross_validation/README.md) |
| 09_object_detection_baseline | 目标检测框回归 Baseline | `detection` | `object_detection` | [09_object_detection_baseline](./09_object_detection_baseline/README.md) |
| 10_segmentation_baseline | 图像分割 Mask Baseline | `segmentation` | `segmentation` | [10_segmentation_baseline](./10_segmentation_baseline/README.md) |
| 11_contrastive_pretraining | 自监督对比学习预训练 | `contrastive` | `representation_learning` | [11_contrastive_pretraining](./11_contrastive_pretraining/README.md) |
| 12_training_report_pack | 训练报告与错误分析 | `report` | `evaluation` | [12_training_report_pack](./12_training_report_pack/README.md) |
