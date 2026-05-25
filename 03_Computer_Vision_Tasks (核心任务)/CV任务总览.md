# CV 任务总览

计算机视觉任务层把基础能力组合成真实问题，包括分类、检测、分割、生成、3D 视觉和跟踪。

## 任务地图

| 任务 | 输入 | 输出 | 典型模型 |
|------|------|------|----------|
| [[./Classification (分类)/图像分类总览|图像分类]] | 图像 | 类别标签 | ResNet、ViT、ConvNeXt |
| [[./Object_Detection/YOLO完整深度解析|目标检测]] | 图像 | 类别 + 边界框 | YOLO、Faster R-CNN、DETR |
| [[./Segmentation (分割：语义、实例、全景)/图像分割总览|图像分割]] | 图像 | 像素级标签或实例 mask | U-Net、DeepLab、Mask R-CNN、SAM |
| [[./Generative_Models (Diffusion, GAN)/生成模型总览|生成模型]] | 噪声/条件/文本 | 图像或视频 | GAN、VAE、Diffusion |
| [[./3D_Vision (NeRF, 3DGS, 点云)/3D视觉总览|3D 视觉]] | 图像/点云/相机参数 | 深度、姿态、三维结构 | SfM、PointNet、NeRF、3DGS |

## 学习建议

1. 从分类理解 backbone。
2. 进入检测理解多尺度、边界框和 NMS。
3. 进入分割理解像素级预测。
4. 进入 3D 视觉理解几何约束和相机模型。
5. 进入生成模型理解概率建模和图像分布。

## 共用基础

- 特征提取：CNN / Transformer backbone。
- 多尺度：FPN、PAN、金字塔、stride。
- 损失函数：交叉熵、IoU、Dice、对比损失。
- 数据增强：裁剪、缩放、颜色扰动、混合增强。
- 评估指标：Accuracy、mAP、mIoU、FID、Chamfer Distance。

## 任务训练入口

- [[./Object_Detection/目标检测训练配置|目标检测训练配置]]
