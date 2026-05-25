# Kalman 滤波与目标跟踪

## 概述

Kalman 滤波是估计系统状态的最优线性滤波器，广泛应用于目标跟踪。它通过预测和测量的加权融合，在有噪声的观测中提取目标状态。

## 基本原理

### 问题设定

假设我们要跟踪一个移动目标（如视频中的汽车）：
- **真实状态**：位置、速度等（但我们看不到）
- **观测值**：检测器给出的位置（有噪声）

**目标**：从嘈杂的观测中估计真实状态。

### 线性系统模型

**状态方程**（系统如何演变）：
$$x_k = A x_{k-1} + B u_k + w_k$$

其中：
- $x_k$：第 k 时刻的状态 [x, y, vx, vy]
- $A$：状态转移矩阵
- $u_k$：控制输入（通常为 0）
- $w_k$：过程噪声 ~ $N(0, Q)$

**观测方程**（我们如何观察状态）：
$$z_k = H x_k + v_k$$

其中：
- $z_k$：观测值 [x_obs, y_obs]
- $H$：观测矩阵
- $v_k$：测量噪声 ~ $N(0, R)$

### 具体例子：简单追踪

**状态向量**：$x = [x, y, vx, vy]^T$（位置和速度）

**状态转移**（恒速运动模型）：
$$\begin{bmatrix} x_{k} \\ y_{k} \\ vx_{k} \\ vy_{k} \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x_{k-1} \\ y_{k-1} \\ vx_{k-1} \\ vy_{k-1} \end{bmatrix} + w_k$$

即：
- $x_k = x_{k-1} + vx_{k-1} + w_x$
- $vx_k = vx_{k-1} + w_{vx}$

**观测**（我们只能看到位置）：
$$\begin{bmatrix} x_{obs} \\ y_{obs} \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \\ vx \\ vy \end{bmatrix} + v_k$$

---

## Kalman 滤波算法

### 两个步骤

#### 步骤 1：预测（Prediction）

基于上一时刻的状态预测当前时刻的状态：

$$\hat{x}_k^- = A \hat{x}_{k-1}^+ + B u_{k-1}$$
$$P_k^- = A P_{k-1}^+ A^T + Q$$

其中：
- $\hat{x}_k^-$：先验估计（预测）
- $P_k^-$：先验误差协方差矩阵（预测不确定性）
- $Q$：过程噪声协方差

**直观理解**：
```
上一时刻：我们知道目标在 (100, 100)，速度 (5, 0)
预测当前：目标应该在 (105, 100)
但是我们不太确定（因为有过程噪声）
```

#### 步骤 2：更新（Update）

融合新的观测值更新状态：

$$K_k = P_k^- H^T (H P_k^- H^T + R)^{-1}$$
$$\hat{x}_k^+ = \hat{x}_k^- + K_k (z_k - H \hat{x}_k^-)$$
$$P_k^+ = (I - K_k H) P_k^-$$

其中：
- $K_k$：Kalman 增益（决定预测和观测的权重比）
- $\hat{x}_k^+$：后验估计（最终估计）
- $P_k^+$：后验误差协方差矩阵
- $R$：测量噪声协方差
- $(z_k - H \hat{x}_k^-)$：观测残差（创新）

**直观理解**：
```
预测：目标在 (105, 100)（不太确定）
观测：检测器说目标在 (103, 98)（也不太确定）
融合：目标可能在 (104, 99)（权衡两者）
```

### 完整算法流程

```
初始化: x_0, P_0

for k = 1, 2, ..., T:
    ┌─ 预测阶段
    │  x_k^- = A * x_{k-1}^+ + u
    │  P_k^- = A * P_{k-1}^+ * A^T + Q
    │
    ├─ 观测新数据 z_k
    │
    ├─ 更新阶段
    │  K_k = P_k^- * H^T / (H * P_k^- * H^T + R)
    │  x_k^+ = x_k^- + K_k * (z_k - H * x_k^-)
    │  P_k^+ = (I - K_k * H) * P_k^-
    │
    └─ 输出 x_k^+（目标位置估计）
```

---

## Kalman 增益的直观理解

$$K_k = \frac{P_k^-}{P_k^- + R}$$

（一维情况）

- 如果 $P_k^- \gg R$：预测很不确定，观测很确定 → $K_k \approx 1$（相信观测）
- 如果 $P_k^- \ll R$：预测很确定，观测很不确定 → $K_k \approx 0$（相信预测）
- 如果 $P_k^- \approx R$：两者同样不确定 → $K_k \approx 0.5$（折中）

---

## 实现示例

### 简单的 1D Kalman 滤波器

```python
import numpy as np

class KalmanFilter1D:
    def __init__(self, x0, p0, q, r):
        """
        x0: 初始位置
        p0: 初始位置不确定度
        q: 过程噪声
        r: 测量噪声
        """
        self.x = x0       # 当前估计
        self.p = p0       # 当前不确定度
        self.q = q        # 过程噪声
        self.r = r        # 测量噪声
    
    def predict(self):
        """预测下一步"""
        # 恒速模型：x 不变
        self.x_pred = self.x
        self.p_pred = self.p + self.q
    
    def update(self, z):
        """用观测值更新"""
        # Kalman 增益
        self.k = self.p_pred / (self.p_pred + self.r)
        
        # 更新位置估计
        self.x = self.x_pred + self.k * (z - self.x_pred)
        
        # 更新不确定度
        self.p = (1 - self.k) * self.p_pred
        
        return self.x
    
    def step(self, z):
        """完整一步"""
        self.predict()
        return self.update(z)

# 使用示例
kf = KalmanFilter1D(x0=0, p0=1.0, q=0.01, r=0.1)

measurements = [1.0, 1.2, 1.1, 1.3, 1.2, 1.4]
estimates = [kf.step(z) for z in measurements]

print("测量值", measurements)
print("估计值", estimates)
```

### 2D 目标跟踪的 Kalman 滤波器

```python
import numpy as np

class KalmanFilterTracker:
    def __init__(self, dt=1.0):
        """
        dt: 时间间隔
        """
        self.dt = dt
        
        # 状态：[x, y, vx, vy]
        self.x = np.array([0, 0, 0, 0], dtype=np.float32)
        
        # 状态转移矩阵
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 观测矩阵（只观测位置）
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # 过程噪声协方差
        self.Q = np.eye(4) * 0.01
        
        # 测量噪声协方差
        self.R = np.eye(2) * 0.1
        
        # 状态不确定性
        self.P = np.eye(4)
    
    def predict(self):
        """预测"""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
    
    def update(self, z):
        """更新（z 是 [x_obs, y_obs]）"""
        # 观测预测
        z_pred = self.H @ self.x
        
        # 观测残差
        y = z - z_pred
        
        # 残差协方差
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman 增益
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        self.x = self.x + K @ y
        
        # 更新不确定性
        self.P = (np.eye(4) - K @ self.H) @ self.P
    
    def step(self, z):
        """完整一步"""
        self.predict()
        self.update(z)
        return self.x[:2]  # 返回位置

# 使用
kf = KalmanFilterTracker(dt=1.0)

# 模拟观测
detections = [
    np.array([10, 20]),
    np.array([11, 21]),
    np.array([12, 22]),
    np.array([13, 23]),
]

for det in detections:
    estimated_pos = kf.step(det)
    print(f"观测: {det}, 估计: {estimated_pos}")
```

---

## 参数调节

### $Q$（过程噪声）调整

$Q$ 反映了预测模型的不确定性。

```python
# 高 Q：预测不可靠，更相信观测
self.Q = np.eye(4) * 0.1  # 强烈波动

# 低 Q：预测可靠，更相信预测
self.Q = np.eye(4) * 0.001  # 平稳运动
```

### $R$（测量噪声）调整

$R$ 反映了观测的准确度。

```python
# 高 R：观测不准确，更相信预测
self.R = np.eye(2) * 0.5  # 嘈杂的检测器

# 低 R：观测准确，更相信观测
self.R = np.eye(2) * 0.01  # 准确的检测器
```

### 实验调节法则

```python
# 开始时保守估计
Q = I * 0.001
R = I * 0.1

# 观察结果：
# - 跟踪滞后：增大 Q
# - 跟踪抖动：减小 Q，增大 R
# - 目标丢失：增大 Q
```

---

## 与目标检测结合：MOT（多目标跟踪）

### DeepSORT 框架

结合 CNN 特征和 Kalman 滤波的多目标跟踪。

```
检测 → 特征提取 → 特征匹配 → Kalman 预测 → 跟踪
(YOLO) (ReID网络)  (匹配器)  (运动模型)  (轨迹)
```

### 关联策略

```python
def associate_detections_to_trackers(detections, trackers):
    """
    将检测结果关联到已有的跟踪轨迹
    """
    # 1. Kalman 预测所有跟踪器的位置
    for tracker in trackers:
        tracker.predict()
    
    # 2. 计算成本矩阵（基于 IoU）
    cost_matrix = compute_iou_distance(detections, trackers)
    
    # 3. 匹配（匈牙利算法）
    matched, unmatched_dets, unmatched_trks = hungarian_algorithm(cost_matrix)
    
    # 4. 更新
    for det_idx, trk_idx in matched:
        trackers[trk_idx].update(detections[det_idx])
    
    # 5. 创建新轨迹
    for det_idx in unmatched_dets:
        new_tracker = KalmanFilterTracker()
        new_tracker.update(detections[det_idx])
        trackers.append(new_tracker)
    
    return trackers
```

---

## 扩展模型

### 变加速度模型

```python
# 状态：[x, y, vx, vy, ax, ay]
A = np.array([
    [1, 0, dt, 0, 0.5*dt**2, 0],
    [0, 1, 0, dt, 0, 0.5*dt**2],
    [0, 0, 1, 0, dt, 0],
    [0, 0, 0, 1, 0, dt],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])
```

### 角度跟踪

```python
# 对于旋转目标（如行人姿态）
# 状态：[x, y, vx, vy, theta, omega]
```

---

## 常见问题

### Q1: Kalman 滤波可以处理非线性运动吗？

**不完全可以**。使用扩展 Kalman 滤波（EKF）或无损 Kalman 滤波（UKF）处理非线性。

### Q2: 如何处理目标消失？

```python
if tracker.age > max_age:
    remove_tracker(tracker)  # 移除老化的轨迹
```

### Q3: 如何避免 ID 切换？

使用外观特征（CNN 特征）配合运动模型，见 DeepSORT。

---

## 性能对比

| 方法 | 精度 | 速度 | 复杂度 |
|------|------|------|--------|
| 简单中心点 | 低 | 快 | 低 |
| Kalman 滤波 | 中 | 快 | 中 |
| DeepSORT | 高 | 中 | 高 |
| 图匹配 | 很高 | 慢 | 很高 |

---

## 参考资源

- **论文**：
  - "An Introduction to the Kalman Filter"
  - "Simple Online and Realtime Tracking" (DeepSORT)

- **实现库**：
  - `filterpy`：Python Kalman 滤波库
  - `sort.py`：极简单的多目标跟踪

---

## 关键链接

- [[./YOLO完整深度解析|YOLO 检测]]
- [[../../../05_Projects_&_Code (项目实战)|项目实战]]

---

**学习建议**：从 1D 例子开始理解原理，再扩展到 2D/3D。通过调节参数体会 Kalman 滤波的工作原理。

