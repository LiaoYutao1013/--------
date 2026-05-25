from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


TASKS = [
    {
        "id": "01_from_scratch_classifier",
        "title": "从零训练图像分类器",
        "mode": "from_scratch",
        "vision_task": "classification",
        "objective": "理解参数、损失、梯度下降和训练/验证划分的完整闭环。",
    },
    {
        "id": "02_minibatch_training",
        "title": "Mini-batch 训练与学习率实验",
        "mode": "mini_batch",
        "vision_task": "classification",
        "objective": "观察 batch size 和学习率对收敛速度的影响。",
    },
    {
        "id": "03_frozen_feature_transfer",
        "title": "冻结特征的迁移学习",
        "mode": "frozen_transfer",
        "vision_task": "classification",
        "objective": "模拟固定 backbone，只训练分类头的迁移学习流程。",
    },
    {
        "id": "04_finetune_last_layers",
        "title": "解冻后几层进行微调",
        "mode": "finetune",
        "vision_task": "classification",
        "objective": "比较只训练分类头与小学习率微调 backbone 的差异。",
    },
    {
        "id": "05_augmentation_ablation",
        "title": "数据增强消融实验",
        "mode": "augmentation",
        "vision_task": "classification",
        "objective": "对比无增强、噪声增强、翻转增强对验证集的影响。",
    },
    {
        "id": "06_hyperparameter_search",
        "title": "超参数搜索训练任务",
        "mode": "hyperparameter_search",
        "vision_task": "classification",
        "objective": "用网格搜索比较学习率、正则化和 epoch 数。",
    },
    {
        "id": "07_class_imbalance_training",
        "title": "类别不均衡训练",
        "mode": "class_imbalance",
        "vision_task": "classification",
        "objective": "比较普通损失和类别加权损失在少数类上的表现。",
    },
    {
        "id": "08_cross_validation",
        "title": "K 折交叉验证",
        "mode": "cross_validation",
        "vision_task": "classification",
        "objective": "学习小数据集上更稳健的模型评估方式。",
    },
    {
        "id": "09_object_detection_baseline",
        "title": "目标检测框回归 Baseline",
        "mode": "detection",
        "vision_task": "object_detection",
        "objective": "训练一个合成检测框回归器，并计算 IoU。",
    },
    {
        "id": "10_segmentation_baseline",
        "title": "图像分割 Mask Baseline",
        "mode": "segmentation",
        "vision_task": "segmentation",
        "objective": "生成合成 mask，学习阈值分割、IoU 和 Dice 指标。",
    },
    {
        "id": "11_contrastive_pretraining",
        "title": "自监督对比学习预训练",
        "mode": "contrastive",
        "vision_task": "representation_learning",
        "objective": "通过正负样本相似度理解对比学习的训练目标。",
    },
    {
        "id": "12_training_report_pack",
        "title": "训练报告与错误分析",
        "mode": "report",
        "vision_task": "evaluation",
        "objective": "把训练曲线、混淆矩阵、错误样本和下一步计划汇总成报告。",
    },
]


MAIN_TEMPLATE = r'''from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def make_classification(seed: int, count: int = 160, imbalance: bool = False) -> list[tuple[list[float], int]]:
    rng = random.Random(seed)
    data: list[tuple[list[float], int]] = []
    while len(data) < count:
        x = rng.uniform(-1.5, 1.5)
        y = rng.uniform(-1.5, 1.5)
        texture = math.sin(4 * x) * 0.2 + rng.uniform(-0.08, 0.08)
        label = 1 if x * x + 0.8 * y + texture > 0.45 else 0
        if imbalance and label == 1 and rng.random() < 0.65:
            continue
        data.append(([x, y, x * x, y * y, texture], label))
    return data


def split_data(data: list, ratio: float = 0.75) -> tuple[list, list]:
    cut = int(len(data) * ratio)
    return data[:cut], data[cut:]


def train_logistic(
    train: list[tuple[list[float], int]],
    val: list[tuple[list[float], int]],
    epochs: int,
    lr: float,
    batch_size: int,
    l2: float = 0.0,
    class_weight: dict[int, float] | None = None,
    initial_weights: list[float] | None = None,
) -> dict:
    feature_count = len(train[0][0])
    weights = initial_weights[:] if initial_weights else [0.0] * feature_count
    bias = 0.0
    history = []
    class_weight = class_weight or {0: 1.0, 1: 1.0}

    for epoch in range(epochs):
        total_loss = 0.0
        for start in range(0, len(train), batch_size):
            batch = train[start : start + batch_size]
            grad_w = [0.0] * feature_count
            grad_b = 0.0
            for features, label in batch:
                pred = sigmoid(dot(weights, features) + bias)
                weight = class_weight[label]
                error = (pred - label) * weight
                for index, value in enumerate(features):
                    grad_w[index] += error * value
                grad_b += error
                total_loss += weight * (-(label * math.log(pred + 1e-9) + (1 - label) * math.log(1 - pred + 1e-9)))
            scale = 1 / len(batch)
            for index in range(feature_count):
                weights[index] -= lr * (grad_w[index] * scale + l2 * weights[index])
            bias -= lr * grad_b * scale
        history.append({"epoch": epoch + 1, "loss": total_loss / len(train), "val_accuracy": accuracy(weights, bias, val)})
    return {"weights": weights, "bias": bias, "history": history, "val_accuracy": accuracy(weights, bias, val), "metrics": classification_metrics(weights, bias, val)}


def accuracy(weights: list[float], bias: float, data: list[tuple[list[float], int]]) -> float:
    correct = 0
    for features, label in data:
        pred = 1 if sigmoid(dot(weights, features) + bias) >= 0.5 else 0
        correct += pred == label
    return correct / len(data)


def classification_metrics(weights: list[float], bias: float, data: list[tuple[list[float], int]]) -> dict:
    tp = tn = fp = fn = 0
    for features, label in data:
        pred = 1 if sigmoid(dot(weights, features) + bias) >= 0.5 else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1:
            fp += 1
        else:
            fn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def augment(data: list[tuple[list[float], int]], mode: str, seed: int) -> list[tuple[list[float], int]]:
    rng = random.Random(seed)
    result = list(data)
    for features, label in data:
        x, y, x2, y2, texture = features
        if mode == "noise":
            nx = x + rng.uniform(-0.08, 0.08)
            ny = y + rng.uniform(-0.08, 0.08)
            result.append(([nx, ny, nx * nx, ny * ny, texture + rng.uniform(-0.04, 0.04)], label))
        elif mode == "flip":
            result.append(([-x, y, x2, y2, -texture], label))
    return result


def box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_l = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_r = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(1e-9, area_l + area_r - inter)


def write_artifacts(result: dict, history: list[dict] | None = None) -> None:
    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if history:
        with (artifacts / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)


def run_from_scratch(config: dict) -> dict:
    data = make_classification(config["seed"])
    train, val = split_data(data)
    model = train_logistic(train, val, epochs=50, lr=0.35, batch_size=len(train))
    return {"experiment": "from_scratch_classifier", **model}


def run_mini_batch(config: dict) -> dict:
    data = make_classification(config["seed"])
    train, val = split_data(data)
    runs = {}
    for batch_size in [8, 16, 64]:
        model = train_logistic(train, val, epochs=35, lr=0.28, batch_size=batch_size)
        runs[str(batch_size)] = {"val_accuracy": model["val_accuracy"], "final_loss": model["history"][-1]["loss"]}
    return {"experiment": "mini_batch_training", "runs": runs}


def run_frozen_transfer(config: dict) -> dict:
    data = make_classification(config["seed"])
    frozen_features = [([f[0] + 0.3 * f[2], f[1] - 0.2 * f[4], f[2] + f[3]], label) for f, label in data]
    train, val = split_data(frozen_features)
    model = train_logistic(train, val, epochs=40, lr=0.25, batch_size=24)
    return {"experiment": "frozen_feature_transfer", "frozen_backbone": True, **model}


def run_finetune(config: dict) -> dict:
    data = make_classification(config["seed"])
    train, val = split_data(data)
    head_only = train_logistic(train, val, epochs=25, lr=0.22, batch_size=16, initial_weights=[0.2, -0.1, 0.0, 0.0, 0.0])
    finetuned = train_logistic(train, val, epochs=25, lr=0.08, batch_size=16, initial_weights=head_only["weights"])
    return {"experiment": "finetune_last_layers", "head_only_accuracy": head_only["val_accuracy"], "finetuned_accuracy": finetuned["val_accuracy"], "metrics": finetuned["metrics"]}


def run_augmentation(config: dict) -> dict:
    data = make_classification(config["seed"])
    train, val = split_data(data)
    runs = {}
    for mode in ["none", "noise", "flip"]:
        augmented = train if mode == "none" else augment(train, mode, config["seed"] + 11)
        model = train_logistic(augmented, val, epochs=35, lr=0.26, batch_size=16)
        runs[mode] = {"train_count": len(augmented), "val_accuracy": model["val_accuracy"], "recall": model["metrics"]["recall"]}
    return {"experiment": "augmentation_ablation", "runs": runs}


def run_hyperparameter_search(config: dict) -> dict:
    data = make_classification(config["seed"])
    train, val = split_data(data)
    trials = []
    for lr in [0.12, 0.24, 0.36]:
        for l2 in [0.0, 0.01, 0.05]:
            model = train_logistic(train, val, epochs=30, lr=lr, batch_size=16, l2=l2)
            trials.append({"lr": lr, "l2": l2, "val_accuracy": model["val_accuracy"], "final_loss": model["history"][-1]["loss"]})
    best = max(trials, key=lambda item: (item["val_accuracy"], -item["final_loss"]))
    return {"experiment": "hyperparameter_search", "best": best, "trials": trials}


def run_class_imbalance(config: dict) -> dict:
    data = make_classification(config["seed"], count=180, imbalance=True)
    train, val = split_data(data)
    positives = sum(label for _, label in train)
    negatives = len(train) - positives
    plain = train_logistic(train, val, epochs=45, lr=0.28, batch_size=16)
    weighted = train_logistic(train, val, epochs=45, lr=0.28, batch_size=16, class_weight={0: 1.0, 1: negatives / max(1, positives)})
    return {"experiment": "class_imbalance_training", "train_positive_count": positives, "plain": plain["metrics"], "weighted": weighted["metrics"]}


def run_cross_validation(config: dict) -> dict:
    data = make_classification(config["seed"], count=150)
    fold_size = len(data) // 5
    folds = []
    for fold in range(5):
        val = data[fold * fold_size : (fold + 1) * fold_size]
        train = data[: fold * fold_size] + data[(fold + 1) * fold_size :]
        model = train_logistic(train, val, epochs=30, lr=0.24, batch_size=16)
        folds.append({"fold": fold + 1, "val_accuracy": model["val_accuracy"]})
    mean_acc = sum(item["val_accuracy"] for item in folds) / len(folds)
    return {"experiment": "cross_validation", "folds": folds, "mean_accuracy": mean_acc}


def run_detection(config: dict) -> dict:
    rng = random.Random(config["seed"])
    samples = []
    for _ in range(80):
        cx = rng.uniform(0.2, 0.8)
        cy = rng.uniform(0.2, 0.8)
        w = rng.uniform(0.12, 0.28)
        h = rng.uniform(0.12, 0.30)
        features = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03), w * h, w / h]
        target = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        samples.append((features, target))
    weights = [[0.0 for _ in range(4)] for _ in range(4)]
    bias = [0.0] * 4
    for _ in range(120):
        for features, target in samples[:60]:
            pred = [dot(row, features) + b for row, b in zip(weights, bias)]
            for out in range(4):
                error = pred[out] - target[out]
                for index, value in enumerate(features):
                    weights[out][index] -= 0.08 * error * value
                bias[out] -= 0.08 * error
    ious = []
    for features, target in samples[60:]:
        pred = [dot(row, features) + b for row, b in zip(weights, bias)]
        pred = [max(0.0, min(1.0, value)) for value in pred]
        ious.append(box_iou(pred, target))
    return {"experiment": "object_detection_baseline", "mean_iou": sum(ious) / len(ious), "min_iou": min(ious), "max_iou": max(ious)}


def run_segmentation(config: dict) -> dict:
    rng = random.Random(config["seed"])
    grid = []
    target = []
    for y in range(24):
        row = []
        mask_row = []
        for x in range(24):
            distance = math.sqrt((x - 12) ** 2 + (y - 11) ** 2)
            value = 1.0 - distance / 15 + rng.uniform(-0.08, 0.08)
            row.append(value)
            mask_row.append(1 if distance < 6 else 0)
        grid.append(row)
        target.append(mask_row)
    candidates = []
    for threshold in [0.45, 0.50, 0.55, 0.60]:
        pred = [[1 if value >= threshold else 0 for value in row] for row in grid]
        inter = sum(pred[y][x] and target[y][x] for y in range(24) for x in range(24))
        union = sum(pred[y][x] or target[y][x] for y in range(24) for x in range(24))
        pred_sum = sum(pred[y][x] for y in range(24) for x in range(24))
        target_sum = sum(target[y][x] for y in range(24) for x in range(24))
        dice = 2 * inter / max(1, pred_sum + target_sum)
        candidates.append({"threshold": threshold, "iou": inter / max(1, union), "dice": dice})
    best = max(candidates, key=lambda item: item["iou"])
    return {"experiment": "segmentation_baseline", "best": best, "candidates": candidates}


def run_contrastive(config: dict) -> dict:
    rng = random.Random(config["seed"])
    anchors = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(16)]
    positives = [[x + rng.uniform(-0.05, 0.05), y + rng.uniform(-0.05, 0.05)] for x, y in anchors]
    negatives = list(reversed(anchors))
    margins = []
    for anchor, positive, negative in zip(anchors, positives, negatives):
        pos_dist = math.sqrt(sum((a - p) ** 2 for a, p in zip(anchor, positive)))
        neg_dist = math.sqrt(sum((a - n) ** 2 for a, n in zip(anchor, negative)))
        margins.append(neg_dist - pos_dist)
    return {"experiment": "contrastive_pretraining", "mean_margin": sum(margins) / len(margins), "positive_pair_count": len(positives)}


def run_report(config: dict) -> dict:
    data = make_classification(config["seed"])
    train, val = split_data(data)
    model = train_logistic(train, val, epochs=35, lr=0.25, batch_size=16)
    errors = []
    for index, (features, label) in enumerate(val):
        score = sigmoid(dot(model["weights"], features) + model["bias"])
        pred = 1 if score >= 0.5 else 0
        if pred != label:
            errors.append({"sample": index, "label": label, "pred": pred, "score": score})
    return {"experiment": "training_report_pack", "summary": {"val_accuracy": model["val_accuracy"], "error_count": len(errors)}, "confusion_matrix": model["metrics"], "top_errors": errors[:5], "next_steps": ["检查错误样本分布", "尝试数据增强", "比较更强 backbone"]}


RUNNERS = {
    "from_scratch": run_from_scratch,
    "mini_batch": run_mini_batch,
    "frozen_transfer": run_frozen_transfer,
    "finetune": run_finetune,
    "augmentation": run_augmentation,
    "hyperparameter_search": run_hyperparameter_search,
    "class_imbalance": run_class_imbalance,
    "cross_validation": run_cross_validation,
    "detection": run_detection,
    "segmentation": run_segmentation,
    "contrastive": run_contrastive,
    "report": run_report,
}


def main() -> int:
    config = json.loads((Path(__file__).resolve().parent / "task.json").read_text(encoding="utf-8"))
    result = RUNNERS[config["mode"]](config)
    wrapped = {
        "title": config["title"],
        "mode": config["mode"],
        "vision_task": config["vision_task"],
        "objective": config["objective"],
        "result": result,
    }
    history = result.get("history") if isinstance(result.get("history"), list) else None
    write_artifacts(wrapped, history)
    print(f"task: {config['title']}")
    print(f"mode: {config['mode']}")
    print(f"vision_task: {config['vision_task']}")
    key_metric = result.get("val_accuracy") or result.get("mean_accuracy") or result.get("mean_iou") or result.get("mean_margin") or result.get("best", {}).get("iou")
    if key_metric is not None:
        print(f"key_metric: {key_metric:.4f}")
    print("status: PASS")
    print("artifact: artifacts/result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def make_readme(task: dict) -> str:
    extensions = {
        "classification": ["替换为 PyTorch Dataset/DataLoader", "把逻辑回归替换成 CNN/ResNet", "加入混淆矩阵和错误样本可视化"],
        "object_detection": ["替换成 YOLO 格式标签", "加入 anchor/anchor-free 对比", "使用 mAP 代替单一 IoU"],
        "segmentation": ["替换成 U-Net 或 DeepLab", "加入 Dice Loss 和 BCE Loss 对比", "保存预测 mask 可视化"],
        "representation_learning": ["替换成 SimCLR 风格增强", "加入温度系数和 batch negatives", "用线性探针评估特征"],
        "evaluation": ["接入真实训练日志", "自动生成 Markdown 复盘", "按类别统计错误样本"],
    }
    items = "\n".join(f"- {item}" for item in extensions.get(task["vision_task"], extensions["classification"]))
    return f"""# {task['title']}

## 任务目标

{task['objective']}

## 覆盖内容

- 训练方式：`{task['mode']}`
- 视觉任务：`{task['vision_task']}`
- 数据来源：脚本内合成数据，可独立运行
- 运行产物：`artifacts/result.json`

## 运行

```bash
python main.py
```

## 建议迁移方向

{items}
"""


def make_index() -> str:
    rows = "\n".join(
        f"| {task['id']} | {task['title']} | `{task['mode']}` | `{task['vision_task']}` | [{task['id']}](./{task['id']}/README.md) |"
        for task in TASKS
    )
    return f"""# 模型训练与视觉任务

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
{rows}
"""


def make_runner() -> str:
    ids = [task["id"] for task in TASKS]
    return f'''from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


TASKS = {json.dumps(ids, indent=4, ensure_ascii=False)}


def main() -> int:
    root = Path(__file__).resolve().parent
    failures: list[str] = []
    for task in TASKS:
        print(f"=== Running {{task}} ===", flush=True)
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=root / task,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        print(completed.stdout.strip())
        if completed.returncode != 0:
            failures.append(task)
    if failures:
        print("Failed tasks:")
        for task in failures:
            print(f"- {{task}}")
        return 1
    print(f"All {{len(TASKS)}} training and vision tasks ran successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    for index, task in enumerate(TASKS, start=1):
        task = {**task, "seed": 20260519 + index}
        task_dir = ROOT / task["id"]
        task_dir.mkdir(exist_ok=True)
        (task_dir / "task.json").write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
        (task_dir / "README.md").write_text(make_readme(task), encoding="utf-8")
        (task_dir / "main.py").write_text(MAIN_TEMPLATE, encoding="utf-8")
    (ROOT / "README.md").write_text(make_index(), encoding="utf-8")
    (ROOT / "run_all.py").write_text(make_runner(), encoding="utf-8")
    print(f"Generated {len(TASKS)} training tasks in {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
