# OpenCV 单目标定实战

父级：[[00_相机标定学习地图]] ｜ 参见：[[02_单目标定流程]]、[[04_畸变校正与误差评估]]

以下示例使用棋盘格。`pattern_size` 必须是每行、每列的**内角点数量**，例如 10 x 7 个方格对应 `(9, 6)`。

```python
from pathlib import Path
import cv2
import numpy as np

image_dir = Path("calib_images")
pattern_size = (9, 6)        # (columns, rows) of inner corners
square_size_mm = 20.0

# 标定板坐标：Z=0，单位由 square_size_mm 决定。
object_template = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
object_template[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
object_template *= square_size_mm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
object_points, image_points = [], []
image_size = None

for path in sorted(image_dir.glob("*.png")):
    image = cv2.imread(str(path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        print(f"skip: corners not found: {path.name}")
        continue

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    object_points.append(object_template.copy())
    image_points.append(corners)
    image_size = gray.shape[::-1]  # (width, height)

if len(object_points) < 10:
    raise RuntimeError("Too few valid images; collect more diverse views.")

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, image_size, None, None
)

np.savez("camera_calibration.npz", camera_matrix=K, dist_coeffs=dist,
         image_size=np.array(image_size), rms=rms)
print("RMS:", rms)
print("K:\n", K)
print("dist:", dist.ravel())
```

## 去畸变与视场选择

`getOptimalNewCameraMatrix` 中的 `alpha` 控制保留视场和去黑边的取舍：

- `alpha=0`：尽量裁掉无效区域，输出通常没有黑边。
- `alpha=1`：尽量保留原始视场，边缘可能出现无效像素。

```python
image = cv2.imread("test.png")
h, w = image.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
undistorted = cv2.undistort(image, K, dist, None, new_K)
cv2.imwrite("test_undistorted.png", undistorted)
```

对连续视频，先调用 `cv2.initUndistortRectifyMap` 生成 `map1, map2`，再在循环中用 `cv2.remap`，不要每帧重复计算映射。

## 文件格式建议

保存时不要只保存矩阵。参数文件还应包含：

```yaml
model: pinhole_radtan
image_width: 1920
image_height: 1080
square_size_unit: mm
rms_reprojection_error_px: 0.31
camera_matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
dist_coeffs: [k1, k2, p1, p2, k3]
```

`rvecs` 是 Rodrigues 旋转向量，不是欧拉角；可通过 `cv2.Rodrigues(rvec)` 转为旋转矩阵。
