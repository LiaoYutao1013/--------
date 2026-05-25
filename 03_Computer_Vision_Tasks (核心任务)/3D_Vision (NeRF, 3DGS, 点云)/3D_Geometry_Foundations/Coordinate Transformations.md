
---

# 📔 Note: 坐标系变换 (Coordinate Transformations)

> **Metadata**
> 
> - **Topic**: #3D_Vision #Geometry
>     
> - **Parent**: [[3D_Geometry_Foundations]]
>     
> - **Status**: #Completed
>     

---

## 1. 概述：视觉的“降维打击”

在计算机视觉中，我们需要将三维世界中的物体映射到二维图像上。这个过程涉及四个坐标系的嵌套转换：

1. **世界坐标系 (World)**：客观世界的统一坐标。
    
2. **相机坐标系 (Camera)**：以相机光心为原点的三维空间。
    
3. **图像坐标系 (Image)**：成像平面上的物理单位（如 mm）坐标。
    
4. **像素坐标系 (Pixel)**：我们最终在屏幕上看到的行列索引（单位：pixel）。
    

---

## 2. 变换详解

### 第一步：世界坐标系 $\to$ 相机坐标系 (外参变换)

这是一个**刚体变换**（旋转 + 平移）。假设空间点在世界坐标系为 $\mathbf{P}_w$，在相机坐标系为 $\mathbf{P}_c$：

$$\mathbf{P}_c = \mathbf{R} \mathbf{P}_w + \mathbf{t}$$

- **$\mathbf{R}$** (Rotation)：$3 \times 3$ 旋转矩阵。
    
- **$\mathbf{t}$** (Translation)：$3 \times 1$ 平移向量。
    
- **齐次坐标表示**：
    
    $$\begin{bmatrix} X_c \\ Y_c \\ Z_c \\ 1 \end{bmatrix} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ 0^T & 1 \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$
    

### 第二步：相机坐标系 $\to$ 图像坐标系 (投影变换)

基于**针孔相机模型**的相似三角形原理。设相机焦距为 $f$：

$$x = f \frac{X_c}{Z_c}, \quad y = f \frac{Y_c}{Z_c}$$

此时 $(x, y)$ 的单位依然是物理长度（如毫米）。

### 第三步：图像坐标系 $\to$ 像素坐标系 (离散化)

将物理长度转换为像素点，并处理原点偏移（像素坐标系原点通常在图像左上角，而图像坐标系原点在中心）。

$$u = \frac{x}{dx} + u_0, \quad v = \frac{y}{dy} + v_0$$

- $dx, dy$：每个像素在图像平面上的物理尺寸。
    
- $u_0, v_0$：主点（图像中心）在像素坐标系下的坐标。
    

---

## 3. 最终组合：相机内参矩阵 (Intrinsic Matrix)

将上述步骤二和三合并，我们可以得到内参矩阵 $\mathbf{K}$：

$$Z_c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & u_0 \\ 0 & f_y & v_0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = \mathbf{K} \mathbf{P}_c$$

其中 $f_x = f/dx$，$f_y = f/dy$。

---

## 4. 全局投影公式

综上所述，世界坐标系中的点 $\mathbf{P}_w$ 到像素点 $\mathbf{p}$ 的转换公式为：

$$Z_c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

- **内参 (Intrinsic)**：由相机自身构造决定 ($\mathbf{K}$)，一旦出厂基本固定。
    
- **外参 (Extrinsic)**：由相机摆放的位置和角度决定 ($[\mathbf{R} | \mathbf{t}]$)。
    

---

## 💡 避坑指南 (Obsidian Tips)

- **手性问题**：注意相机坐标系通常符合右手定则（$Z$ 轴指向相机前方）。
    
- **齐次坐标**：为什么要加那个 1？为了把旋转和平移写进一个矩阵，方便 GPU 进行大规模矩阵运算。
    
- **深度 $Z_c$**：在公式左侧的 $Z_c$ 告诉我们，同一个像素点其实对应空间中一条射线上的无数个点——这就是为什么单目相机无法直接获取深度的数学根源。