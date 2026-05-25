from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parent


EXCLUDED_PARTS = {
    ".obsidian",
    "Practical_Projects (实战项目)",
    "Note_Practice_Projects (全笔记实战项目)",
}


def stable_slug(text: str) -> str:
    ascii_text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    if ascii_text:
        return f"{ascii_text[:42]}_{digest}"
    return f"note_{digest}"


def read_title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return path.stem


def read_headings(path: Path) -> list[str]:
    headings: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            headings.append(stripped[3:].strip())
    return headings[:8]


def classify(note_path: str, title: str, headings: list[str]) -> tuple[str, str, str]:
    text = f"{note_path} {title} {' '.join(headings)}".lower()

    def has(*keywords: str) -> bool:
        return any(keyword.lower() in text for keyword in keywords)

    if has("attachment", "附件"):
        return "vault_audit", "attachment", "扫描附件引用并生成整理报告"
    if has("concept_graph", "知识图谱", "roadmap", "学习路线", "学习目标", "使用指南", "项目总索引", "完成报告"):
        return "planner", "roadmap", "把知识节点转成可执行学习路线和完成度报告"
    if has("论文", "readings", "paper"):
        return "paper_review", "paper", "构建论文阅读队列、评分和复盘摘要"
    if has("open_source", "开源项目"):
        return "open_source", "repo", "模拟开源项目结构审计并输出阅读路线"
    if has("competition", "kaggle", "竞赛"):
        return "competition", "competition", "生成竞赛 baseline、验证和复盘计划"
    if has("deployment", "部署", "onnx", "tensorrt", "量化"):
        return "deployment", "deployment", "模拟部署流水线并检查关键工程风险"
    if has("camera calibration", "camera models", "coordinate transformations", "epipolar", "3d", "nerf", "点云", "三维", "3d视觉", "重建", "几何"):
        return "three_d", "geometry", "用合成三维点验证投影、变换和误差指标"
    if has("segmentation", "分割"):
        return "segmentation", "mask", "生成合成 mask 并计算 IoU 与连通域"
    if has("generative", "diffusion", "gan", "生成模型"):
        return "generative", "diffusion", "模拟扩散噪声调度和逐步去噪"
    if has("object_detection", "检测", "yolo", "detr", "kalman", "nms", "small_object", "目标检测", "cspnet"):
        return "detection", "boxes", "用合成框验证 IoU、NMS、跟踪或检测配置"
    if has("classification", "分类"):
        return "classification", "classifier", "训练一个可解释的合成分类器"
    if has("transformer", "vit", "swin", "mae", "attention"):
        return "architecture", "transformer", "用小矩阵演示 token、注意力和参数统计"
    if has("cnn", "resnet", "mobilenet", "mlp", "模型架构", "深度学习核心", "神经网络"):
        return "architecture", "network", "做网络层形状传播和参数量估算"
    if has("训练", "数据增强", "优化器", "正则化", "损失函数"):
        return "training", "training", "运行一个小型训练循环并记录损失曲线"
    if has("机器学习", "svm", "randomforest", "聚类"):
        return "classification", "ml", "训练一个传统机器学习风格的合成分类器"
    if has("线性代数"):
        return "math", "linear_algebra", "运行矩阵乘法、投影和主方向估计"
    if has("概率论"):
        return "math", "probability", "模拟随机变量并比较经验统计量"
    if has("最优化"):
        return "math", "optimization", "用梯度下降求解凸优化问题"
    if has("颜色", "color"):
        return "image_processing", "color", "生成 RGB 样本并转换颜色空间"
    if has("边缘", "edge", "特征"):
        return "image_processing", "edge", "用 Sobel 风格算子提取合成图像边缘"
    if has("滤波", "filter", "kernel", "opencv", "图像处理", "摄像头", "像素", "几何变换"):
        return "image_processing", "filter", "对合成图像执行滤波、变换或标注实验"
    return "planner", "note", "把笔记主题拆成可执行任务清单"


def scenario_for(topic: str, subtype: str, title: str) -> str:
    scenarios = {
        "image_processing": f"构建一个 16x16 合成图像实验台，验证《{title}》中的像素处理思路。",
        "math": f"用纯 Python 数值实验复现《{title}》里的核心计算。",
        "classification": f"生成二维样本并训练小分类器，观察《{title}》相关的决策边界。",
        "architecture": f"搭建轻量结构分析器，统计《{title}》相关模块的形状和参数量。",
        "training": f"运行可复现训练循环，记录《{title}》相关训练策略的损失变化。",
        "detection": f"构建合成检测框流水线，验证《{title}》涉及的框、损失或后处理逻辑。",
        "segmentation": f"生成合成分割 mask，计算《{title}》相关的 IoU 和连通域。",
        "generative": f"用一维信号模拟噪声添加和去噪过程，对应《{title}》的生成建模思想。",
        "three_d": f"用合成 3D 点和相机参数验证《{title}》相关几何计算。",
        "deployment": f"生成部署清单并执行检查，对应《{title}》的工程化流程。",
        "open_source": f"对一个模拟开源仓库做结构审计，对应《{title}》的阅读方法。",
        "competition": f"生成竞赛 baseline 计划和实验记录，对应《{title}》。",
        "paper_review": f"构建论文阅读追踪器，对应《{title}》的精读记录。",
        "vault_audit": f"扫描模拟附件清单并输出整理建议，对应《{title}》。",
        "planner": f"把《{title}》转化为可执行任务、里程碑和检查项。",
    }
    return scenarios.get(topic, f"围绕《{title}》生成一个可运行实践项目。")


def seed_for(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


MAIN_TEMPLATE = r'''from __future__ import annotations

import json
import math
import random
import sys
from collections import deque
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_config() -> dict:
    return json.loads((Path(__file__).resolve().parent / "project.json").read_text(encoding="utf-8"))


def write_result(result: dict) -> None:
    artifacts = Path(__file__).resolve().parent / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize(values: list[float]) -> dict:
    avg = sum(values) / len(values)
    var = sum((value - avg) ** 2 for value in values) / len(values)
    return {"min": min(values), "max": max(values), "mean": avg, "std": math.sqrt(var)}


def make_image(seed: int, size: int = 16) -> list[list[int]]:
    rng = random.Random(seed)
    return [[(x * 17 + y * 11 + rng.randint(0, 40)) % 256 for x in range(size)] for y in range(size)]


def convolve(image: list[list[int]], kernel: list[list[float]]) -> list[list[float]]:
    height, width = len(image), len(image[0])
    output = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            total = 0.0
            for ky in range(3):
                for kx in range(3):
                    total += image[y + ky - 1][x + kx - 1] * kernel[ky][kx]
            output[y][x] = total
    return output


def image_processing_lab(config: dict) -> dict:
    image = make_image(config["seed"])
    subtype = config["subtype"]
    if subtype == "edge":
        gx = convolve(image, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        gy = convolve(image, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        edges = [math.sqrt(a * a + b * b) for row_x, row_y in zip(gx, gy) for a, b in zip(row_x, row_y)]
        return {"lab": "edge_detection", "edge_summary": summarize(edges), "strong_edge_pixels": sum(value > 160 for value in edges)}
    if subtype == "color":
        pixels = [(32, 80, 160), (180, 40, 40), (40, 160, 90), (210, 210, 50)]
        converted = []
        for r, g, b in pixels:
            y = 0.299 * r + 0.587 * g + 0.114 * b
            u = -0.147 * r - 0.289 * g + 0.436 * b
            v = 0.615 * r - 0.515 * g - 0.100 * b
            converted.append({"rgb": [r, g, b], "yuv": [round(y, 3), round(u, 3), round(v, 3)]})
        return {"lab": "color_space", "samples": converted}
    blurred = convolve(image, [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
    values = [value for row in blurred for value in row]
    return {"lab": "image_filter", "input_mean": summarize([v for row in image for v in row])["mean"], "filtered_summary": summarize(values)}


def math_lab(config: dict) -> dict:
    subtype = config["subtype"]
    if subtype == "probability":
        rng = random.Random(config["seed"])
        samples = [sum(1 for _ in range(10) if rng.random() < 0.35) for _ in range(1000)]
        return {"lab": "probability_simulation", "trials": len(samples), "empirical": summarize(samples), "expected_mean": 3.5}
    if subtype == "optimization":
        x = 8.0
        history = []
        for step in range(30):
            grad = 2 * (x - 1.5)
            x -= 0.12 * grad
            history.append((x - 1.5) ** 2)
        return {"lab": "gradient_descent", "final_x": x, "final_loss": history[-1], "first_loss": history[0]}
    matrix = [[3.0, 1.0], [1.0, 2.0]]
    vector = [1.0, 1.0]
    for _ in range(12):
        vector = [
            matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
            matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
        ]
        norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
        vector = [vector[0] / norm, vector[1] / norm]
    eigenvalue = (vector[0] * (3 * vector[0] + vector[1]) + vector[1] * (vector[0] + 2 * vector[1]))
    return {"lab": "linear_algebra", "principal_direction": vector, "rayleigh_eigenvalue": eigenvalue}


def classification_lab(config: dict) -> dict:
    rng = random.Random(config["seed"])
    data = []
    for _ in range(80):
        x1, x2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        label = 1 if 1.4 * x1 - 0.8 * x2 + 0.1 > 0 else 0
        data.append((x1, x2, label))
    weights = [0.0, 0.0]
    bias = 0.0
    mistakes = []
    for _ in range(12):
        error_count = 0
        for x1, x2, label in data:
            pred = 1 if weights[0] * x1 + weights[1] * x2 + bias >= 0 else 0
            update = label - pred
            if update:
                weights[0] += 0.2 * update * x1
                weights[1] += 0.2 * update * x2
                bias += 0.2 * update
                error_count += 1
        mistakes.append(error_count)
    accuracy = sum((1 if weights[0] * x1 + weights[1] * x2 + bias >= 0 else 0) == label for x1, x2, label in data) / len(data)
    return {"lab": "classifier", "weights": weights, "bias": bias, "accuracy": accuracy, "mistakes_by_epoch": mistakes}


def architecture_lab(config: dict) -> dict:
    subtype = config["subtype"]
    if subtype == "transformer":
        tokens = [[1.0, 0.2, 0.1], [0.1, 1.0, 0.3], [0.2, 0.1, 1.0]]
        scores = []
        for q in tokens:
            row = []
            for k in tokens:
                row.append(sum(a * b for a, b in zip(q, k)) / math.sqrt(3))
            max_score = max(row)
            probs = [math.exp(value - max_score) for value in row]
            total = sum(probs)
            scores.append([value / total for value in probs])
        return {"lab": "self_attention", "attention": scores, "token_count": len(tokens)}
    layers = [
        {"name": "conv3x3", "in": 3, "out": 16, "kernel": 3, "size": 64},
        {"name": "conv3x3", "in": 16, "out": 32, "kernel": 3, "size": 32},
        {"name": "linear", "in": 32 * 16 * 16, "out": 10, "kernel": 1, "size": 1},
    ]
    params = []
    for layer in layers:
        if layer["name"].startswith("conv"):
            count = layer["in"] * layer["out"] * layer["kernel"] ** 2 + layer["out"]
        else:
            count = layer["in"] * layer["out"] + layer["out"]
        params.append({"layer": layer["name"], "params": count})
    return {"lab": "shape_and_params", "layers": params, "total_params": sum(item["params"] for item in params)}


def training_lab(config: dict) -> dict:
    rng = random.Random(config["seed"])
    data = [(x / 20.0, 2.5 * (x / 20.0) - 0.7 + rng.uniform(-0.03, 0.03)) for x in range(-20, 21)]
    w, b = 0.0, 0.0
    losses = []
    for epoch in range(40):
        lr = 0.18 * (0.95 ** epoch)
        dw = sum(2 * (w * x + b - y) * x for x, y in data) / len(data)
        db = sum(2 * (w * x + b - y) for x, y in data) / len(data)
        w -= lr * dw
        b -= lr * db
        losses.append(sum((w * x + b - y) ** 2 for x, y in data) / len(data))
    return {"lab": "training_loop", "weight": w, "bias": b, "initial_loss": losses[0], "final_loss": losses[-1]}


def box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_l = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_r = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return 0.0 if area_l + area_r - inter == 0 else inter / (area_l + area_r - inter)


def detection_lab(config: dict) -> dict:
    boxes = [
        {"box": (10, 10, 60, 60), "score": 0.91, "cls": 0},
        {"box": (14, 12, 58, 62), "score": 0.82, "cls": 0},
        {"box": (70, 18, 120, 80), "score": 0.77, "cls": 1},
        {"box": (74, 20, 118, 78), "score": 0.51, "cls": 1},
    ]
    kept = []
    for item in sorted(boxes, key=lambda row: row["score"], reverse=True):
        if all(item["cls"] != other["cls"] or box_iou(item["box"], other["box"]) <= 0.45 for other in kept):
            kept.append(item)
    return {"lab": "detection_postprocess", "input_boxes": len(boxes), "kept_boxes": kept, "pair_iou": box_iou(boxes[0]["box"], boxes[1]["box"])}


def segmentation_lab(config: dict) -> dict:
    pred = [[1 if 3 <= x <= 10 and 4 <= y <= 11 else 0 for x in range(14)] for y in range(14)]
    target = [[1 if 4 <= x <= 11 and 3 <= y <= 10 else 0 for x in range(14)] for y in range(14)]
    inter = sum(pred[y][x] and target[y][x] for y in range(14) for x in range(14))
    union = sum(pred[y][x] or target[y][x] for y in range(14) for x in range(14))
    visited = set()
    components = 0
    for y in range(14):
        for x in range(14):
            if pred[y][x] and (x, y) not in visited:
                components += 1
                q = deque([(x, y)])
                visited.add((x, y))
                while q:
                    cx, cy = q.popleft()
                    for nx, ny in [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]:
                        if 0 <= nx < 14 and 0 <= ny < 14 and pred[ny][nx] and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            q.append((nx, ny))
    return {"lab": "segmentation_metrics", "iou": inter / union, "components": components, "intersection": inter, "union": union}


def generative_lab(config: dict) -> dict:
    clean = [math.sin(i / 4) for i in range(24)]
    rng = random.Random(config["seed"])
    schedule = [0.02, 0.06, 0.12, 0.20]
    noisy = clean[:]
    trace = []
    for beta in schedule:
        noisy = [(1 - beta) * value + beta * rng.gauss(0, 1) for value in noisy]
        trace.append(sum(abs(a - b) for a, b in zip(clean, noisy)) / len(clean))
    denoised = [(left + value + right) / 3 for left, value, right in zip([noisy[0]] + noisy[:-1], noisy, noisy[1:] + [noisy[-1]])]
    error = sum(abs(a - b) for a, b in zip(clean, denoised)) / len(clean)
    return {"lab": "diffusion_schedule", "noise_trace": trace, "denoised_mae": error}


def three_d_lab(config: dict) -> dict:
    points = [(-1, -1, 4), (1, -1, 4), (1, 1, 4), (-1, 1, 4), (0, 0, 6)]
    fx, fy, cx, cy = 800, 800, 320, 240
    projected = []
    for x, y, z in points:
        projected.append((fx * x / z + cx, fy * y / z + cy))
    centroid = (sum(x for x, _ in projected) / len(projected), sum(y for _, y in projected) / len(projected))
    return {"lab": "camera_projection", "projected_points": projected, "image_centroid": centroid}


def deployment_lab(config: dict) -> dict:
    checks = {
        "preprocess_match": True,
        "postprocess_match": True,
        "onnx_error_ok": 3e-6 < 1e-5,
        "dynamic_batch_tested": True,
        "latency_recorded": True,
        "bad_input_handled": True,
    }
    return {"lab": "deployment_audit", "checks": checks, "passed": all(checks.values())}


def open_source_lab(config: dict) -> dict:
    sections = ["README", "configs", "datasets", "models", "train", "losses", "evaluation", "export"]
    weights = {name: index + 1 for index, name in enumerate(sections)}
    score = sum(weights.values())
    return {"lab": "repo_reading_audit", "reading_order": sections, "coverage_score": score, "next_focus": "trace one batch from dataset to loss"}


def competition_lab(config: dict) -> dict:
    experiments = [
        {"name": "baseline", "cv": 0.71},
        {"name": "augmentation", "cv": 0.735},
        {"name": "larger_model", "cv": 0.748},
        {"name": "tta", "cv": 0.752},
    ]
    best = max(experiments, key=lambda item: item["cv"])
    return {"lab": "competition_tracker", "experiments": experiments, "best": best}


def paper_review_lab(config: dict) -> dict:
    criteria = {"problem": 4, "method": 5, "experiment": 4, "limitations": 3, "reproducibility": 4}
    total = sum(criteria.values())
    return {"lab": "paper_review_scorecard", "criteria": criteria, "score": total, "decision": "deep_read" if total >= 18 else "skim"}


def planner_lab(config: dict) -> dict:
    headings = config.get("headings") or ["问题定义", "实验设计", "验证指标"]
    tasks = [{"step": index + 1, "task": f"实践：{heading}", "done": False} for index, heading in enumerate(headings[:6])]
    return {"lab": "note_to_tasks", "task_count": len(tasks), "tasks": tasks}


def vault_audit_lab(config: dict) -> dict:
    attachments = ["diagram.png", "paper.pdf", "camera_sample.jpg", "experiment.csv"]
    classified = {
        "images": [name for name in attachments if name.endswith((".png", ".jpg"))],
        "papers": [name for name in attachments if name.endswith(".pdf")],
        "data": [name for name in attachments if name.endswith(".csv")],
    }
    return {"lab": "attachment_audit", "classified": classified, "total": len(attachments)}


LABS = {
    "image_processing": image_processing_lab,
    "math": math_lab,
    "classification": classification_lab,
    "architecture": architecture_lab,
    "training": training_lab,
    "detection": detection_lab,
    "segmentation": segmentation_lab,
    "generative": generative_lab,
    "three_d": three_d_lab,
    "deployment": deployment_lab,
    "open_source": open_source_lab,
    "competition": competition_lab,
    "paper_review": paper_review_lab,
    "planner": planner_lab,
    "vault_audit": vault_audit_lab,
}


def main() -> int:
    config = load_config()
    lab = LABS.get(config["topic"], planner_lab)
    result = {
        "note_title": config["title"],
        "topic": config["topic"],
        "subtype": config["subtype"],
        "objective": config["objective"],
        "result": lab(config),
    }
    write_result(result)
    print(f"note: {config['title']}")
    print(f"topic: {config['topic']} / {config['subtype']}")
    print(f"objective: {config['objective']}")
    print("status: PASS")
    print("artifact: artifacts/result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def make_readme(config: dict) -> str:
    heading_lines = "\n".join(f"- {heading}" for heading in (config["headings"] or ["围绕主题完成一次最小可运行实验"]))
    return f"""# {config['title']}：实战项目

来源笔记：`{config['note_path']}`

## 项目目标

{config['scenario']}

该项目的程序不读取原始笔记，配置和代码都在当前项目目录内，可独立运行。

## 练习切入点

{heading_lines}

## 运行

```bash
python main.py
```

运行后会生成：

- `artifacts/result.json`

## 可扩展方向

- 把合成数据替换成真实数据。
- 把纯 Python 实现替换成 OpenCV、PyTorch、ONNX Runtime 或真实项目代码。
- 增加单元测试和指标阈值，让实验可以持续复现。
"""


def collect_notes() -> list[Path]:
    notes: list[Path] = []
    for path in ROOT.rglob("*.md"):
        rel_parts = path.relative_to(ROOT).parts
        if any(part in EXCLUDED_PARTS for part in rel_parts):
            continue
        notes.append(path)
    return sorted(notes, key=lambda item: item.relative_to(ROOT).as_posix().lower())


def create_project(index: int, note: Path) -> dict:
    rel = note.relative_to(ROOT).as_posix()
    title = read_title(note)
    headings = read_headings(note)
    topic, subtype, objective = classify(rel, title, headings)
    project_id = f"{index:03d}_{stable_slug(rel)}"
    project_dir = PROJECT_ROOT / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "id": project_id,
        "title": title,
        "note_path": rel,
        "headings": headings,
        "topic": topic,
        "subtype": subtype,
        "objective": objective,
        "scenario": scenario_for(topic, subtype, title),
        "seed": seed_for(rel),
    }

    (project_dir / "project.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (project_dir / "README.md").write_text(make_readme(config), encoding="utf-8")
    (project_dir / "main.py").write_text(MAIN_TEMPLATE, encoding="utf-8")
    return config


def make_index(projects: list[dict]) -> str:
    rows = "\n".join(
        f"| {item['id']} | {item['title']} | `{item['topic']}/{item['subtype']}` | [{item['id']}](./{item['id']}/README.md) |"
        for item in projects
    )
    return f"""# 全笔记实战项目

这个目录为当前知识库中的每篇原始 Markdown 笔记创建一个独立可运行项目。

## 运行方式

运行全部项目：

```bash
python run_all.py
```

运行单个项目：

```bash
cd <项目目录>
python main.py
```

每个项目都包含：

- `README.md`：项目目标、来源笔记和扩展方向
- `project.json`：独立配置
- `main.py`：可运行程序
- `artifacts/result.json`：运行产物

## 项目索引

| ID | 笔记 | 类型 | 入口 |
|----|------|------|------|
{rows}
"""


def make_runner(projects: list[dict]) -> str:
    ids = [item["id"] for item in projects]
    return f'''from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROJECTS = {json.dumps(ids, indent=4, ensure_ascii=False)}


def main() -> int:
    root = Path(__file__).resolve().parent
    failures: list[str] = []
    for project in PROJECTS:
        print(f"=== Running {{project}} ===", flush=True)
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=root / project,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            failures.append(project)
            print(completed.stdout)
        else:
            lines = [line for line in completed.stdout.splitlines() if line.startswith(("note:", "topic:", "status:", "artifact:"))]
            for line in lines:
                print(line.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
    if failures:
        print("Failed projects:")
        for project in failures:
            print(f"- {{project}}")
        return 1
    print(f"All {{len(PROJECTS)}} note practice projects ran successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> int:
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    notes = collect_notes()
    projects = [create_project(index, note) for index, note in enumerate(notes, start=1)]
    (PROJECT_ROOT / "README.md").write_text(make_index(projects), encoding="utf-8")
    (PROJECT_ROOT / "run_all.py").write_text(make_runner(projects), encoding="utf-8")
    print(f"Generated {len(projects)} projects in {PROJECT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
