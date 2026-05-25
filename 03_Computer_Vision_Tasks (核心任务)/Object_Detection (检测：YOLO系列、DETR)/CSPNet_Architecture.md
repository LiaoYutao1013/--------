# CSPNet 架构：跨阶段部分连接

> 完整笔记已维护在：[[../Object_Detection/CSPNet_Architecture|CSPNet_Architecture]]

## 快速理解

**CSPNet**（Cross Stage Partial Network）是一种 CNN stage 级结构设计，常见于 YOLOv4 的 CSPDarknet-53、YOLOv5 的 C3、YOLOv8 的 C2f 等模块。

核心思想：把输入特征分成两路，一路经过较重的 bottleneck/residual 变换，另一路走较短路径，最后通过 Concat 和 1×1 Conv 融合。

```text
Input X
├─ heavy branch: 1×1 Conv -> Bottleneck × n
└─ light branch: 1×1 Conv / shortcut
        ↓
     Concat
        ↓
     1×1 Conv
        ↓
     Output
```

## 为什么有效

- **减少冗余计算**：不是所有通道都经过完整的重型变换。
- **保留梯度路径**：旁路分支让梯度更容易跨 stage 流动。
- **提高实时检测效率**：速度、显存和精度之间更容易取得平衡。
- **便于模块化演化**：后续 C3、C2f、ELAN 都继承了“分支 + 拼接 + 融合”的思想。

## 与 ResNet 的区别

| 对比项 | ResNet | CSPNet |
|--------|--------|--------|
| 融合方式 | block 内加法 | stage 内拼接后卷积 |
| 计算路径 | 大部分特征走完整主干 | 部分特征走重变换，部分走短路径 |
| 优势 | 训练深层网络稳定 | 降低冗余，适合实时检测 |
| 典型应用 | 分类骨干 | YOLO 系列骨干和 neck |

## 学习重点

1. 先理解 stage 级通道分流，不要把 CSP 简化理解成“残差块切一半”。
2. 对比 `CSPDarknet -> C3 -> C2f` 的演化，关注特征流和梯度路径如何变化。
3. 结合 YOLO 的 neck/head 看 CSP 的价值：它提供更高效的多尺度特征输入。

## 相关链接

- [[../Object_Detection/CSPNet_Architecture|完整 CSPNet 笔记]]
- [[./YOLO 算法原理全深度解析 (Master Deep Dive)|YOLO 算法原理全深度解析]]
- [[./YOLO_Loss_Functions|YOLO 损失函数]]
