from __future__ import annotations

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
