# SVM 支持向量机

SVM（Support Vector Machine）是一类最大间隔分类模型。它希望找到一个决策边界，使两类样本之间的间隔尽可能大。

## 直觉

对二分类问题，线性分类器的决策函数为：

$$
f(x)=w^Tx+b
$$

分类规则：

$$
\hat{y}=\text{sign}(w^Tx+b)
$$

SVM 不只要求分类正确，还希望离决策边界最近的样本尽可能远。

## 最大间隔

硬间隔 SVM 的目标：

$$
\min_{w,b}\frac{1}{2}\|w\|^2
$$

约束：

$$
y_i(w^Tx_i+b)\ge 1
$$

间隔大小与 $\|w\|$ 成反比，所以最小化 $\|w\|$ 等价于最大化间隔。

## 软间隔

真实数据通常不可完全线性分开，需要允许少量错误：

$$
\min_{w,b}\frac{1}{2}\|w\|^2+C\sum_i\xi_i
$$

其中：

- $C$ 越大：越重视训练集分类正确，可能过拟合。
- $C$ 越小：越允许错分，间隔更宽，可能欠拟合。

## Hinge Loss

SVM 常用 hinge loss：

$$
\max(0, 1-y_i(w^Tx_i+b))
$$

如果样本被正确分类且间隔足够大，损失为 0。

## 核函数

当数据非线性可分时，可以用核函数隐式映射到高维空间。

| 核函数 | 适用场景 |
|--------|----------|
| linear | 高维稀疏特征、大样本、可解释 baseline |
| poly | 有明确多项式交互关系 |
| rbf | 常用非线性核，适合中小规模数据 |
| sigmoid | 类似神经网络激活，但较少作为首选 |

RBF 核常见参数：

- `C`：错分惩罚。
- `gamma`：单个样本影响范围。`gamma` 越大，边界越复杂。

## 示例代码：线性 SVM

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

X, y = make_classification(
    n_samples=600,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    random_state=0,
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=1.0, max_iter=5000)),
])

model.fit(X_train, y_train)
pred = model.predict(X_val)

print(classification_report(y_val, pred))
```

## 示例代码：RBF SVM

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=400, noise=0.18, random_state=0)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10.0, gamma=0.5)),
])

model.fit(X_train, y_train)
pred = model.predict(X_val)

print("accuracy:", accuracy_score(y_val, pred))
```

## 示例代码：网格搜索调参

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_classification(
    n_samples=500,
    n_features=12,
    n_informative=5,
    random_state=0,
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC()),
])

param_grid = {
    "svm__kernel": ["linear", "rbf"],
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", 0.1, 1.0],
}

search = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
)

search.fit(X_train, y_train)

print("best params:", search.best_params_)
print("best cv score:", search.best_score_)
print("val score:", search.score(X_val, y_val))
```

## 示例代码：HOG + SVM 图像分类骨架

```python
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def hog_feature(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 128))

    hog = cv2.HOGDescriptor(
        _winSize=(64, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    return hog.compute(image).ravel()

# data/person/*.jpg
# data/background/*.jpg
paths = []
labels = []
for name, label in [("person", 1), ("background", 0)]:
    for path in Path("data").joinpath(name).glob("*.jpg"):
        paths.append(path)
        labels.append(label)

X = np.array([hog_feature(path) for path in paths])
y = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=0.5, max_iter=5000)),
])

model.fit(X_train, y_train)
pred = model.predict(X_val)
print(classification_report(y_val, pred))
```

## SVM 在 CV 中的典型位置

```text
图像
  ↓
HOG / SIFT / LBP / 颜色直方图
  ↓
SVM 分类器
  ↓
类别或目标存在性
```

经典例子：

- HOG + Linear SVM 行人检测。
- SIFT BoVW + SVM 图像分类。
- 小样本缺陷分类 baseline。

## 优缺点

优点：

- 小样本和中等维度特征上很强。
- 最大间隔带来较好的泛化直觉。
- 核函数可以处理非线性边界。

局限：

- 大数据集训练成本高。
- RBF 核参数敏感。
- 原始图像通常不能直接输入，需要特征工程。
- 概率输出不是原生目标，需要 `probability=True` 或校准。
