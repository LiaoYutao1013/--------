# 📖 机器视觉笔记：线性滤波器 (Linear Filters)

## 1. 概述

**线性滤波器**是图像处理中最基础的工具之一。其核心思想是：输出图像的每个像素值是输入像素邻域内像素值的**加权线性组合**。

- **用途**：去噪（平滑）、边缘检测（锐化）、特征提取。
    
- **数学基础**：卷积 (Convolution) 与 相关 (Correlation)。
    

---

## 2. 核心数学原理

### 2.1 互相关 (Cross-Correlation)

互相关描述了**算子（核）**在图像上滑动并计算乘积和的过程。

对于图像 $I$ 和核 $K$，在位置 $(i, j)$ 的响应为：

$$(G \otimes I)(i,j) = \sum_{u=-k}^k \sum_{v=-k}^k K(u,v) I(i+u, j+v)$$

### 2.2 卷积 (Convolution)

卷积与互相关类似，但在计算前需要将核进行**180度翻转**。

$$(G * I)(i,j) = \sum_{u=-k}^k \sum_{v=-k}^k K(u,v) I(i-u, j-v)$$

> [!TIP]
> 
> 在大多数深度学习框架（如 PyTorch）和机器视觉库（如 OpenCV）中，所谓的“卷积层”实际上通常实现的是**互相关**，因为对于对称核（如高斯核），两者结果一致。

---

## 3. 常见线性滤波器分类

### A. 平滑/低通滤波器 (Smoothing/Low-pass)

用于去除高频噪声，使图像变模糊。

1. **均值滤波 (Box Filter)**：
    
    - **核特性**：所有元素相等且总和为 1。
        
    - **效果**：简单粗暴，容易产生“鬼影”现象，不保留边缘。
        
2. **高斯滤波 (Gaussian Filter)**：
    
    - **公式**：$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$
        
    - **特点**：权重随中心距离呈正态分布。它是唯一的**旋转对称**且**可分离**的平滑滤波器。
        
    - **应用**：预处理去噪的首选。
        

---

### B. 锐化/高通滤波器 (Sharpening/High-pass)

用于提取图像的边缘和细节（提取高频成分）。

1. **Sobel 算子**：
    
    - 计算水平和垂直梯度。
        
    - $K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$，$K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$
        
2. **拉普拉斯算子 (Laplacian)**：
    
    - 二阶导数算子，对突变非常敏感。
        
    - 常用于边缘增强。
        

---

## 4. 关键特性

### 4.1 线性与平移不变性 (LSI)

- **线性**：$Filter(aI_1 + bI_2) = a \cdot Filter(I_1) + b \cdot Filter(I_2)$
    
- **平移不变性**：如果图像平移，滤波结果也随之平移。
    

### 4.2 可分离性 (Separability)

如果一个 2D 核可以表示为两个 1D 核的乘积（外积），则称其为可分离的。

- **例子**：高斯核。
    
- **优势**：将 $M \times N$ 的计算量降低为 $M + N$。
    
    > _示例：$K_{2D} = v \cdot h^T$。先进行行滤波，再进行列滤波。_
    

---

## 5. 边界处理 (Boundary Conditions)

当算子覆盖图像边缘时，需要对“越界”像素进行填充：

- **Zero-padding**：补 0（产生黑边）。
    
- **Wrap-around**：周期性循环（图像右边接左边）。
    
- **Reflection**：镜像反射（如 `abc | cba`），**最常用**，过渡自然。
    
- **Clamp/Replicate**：复制边缘像素。
    

---

## 6. OpenCV 常用函数 (Python)

Python

```
import cv2
import numpy as np

# 1. 高斯模糊
blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)

# 2. 自定义算子 (使用 filter2D)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) # 简单锐化核
sharpened = cv2.filter2D(img, -1, kernel)

# 3. Sobel 边缘检测
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
```

---

## 7. 关联学习路径

- [[非线性滤波器]] (如 中值滤波、双边滤波)
    
- [[傅里叶变换与频域滤波]]
    
- [[Canny边缘检测算法]]
    
- [[卷积神经网络CNN基础]]
    

---

**Tags**: #MachineVision #ImageProcessing #LinearFilters #SignalProcessing