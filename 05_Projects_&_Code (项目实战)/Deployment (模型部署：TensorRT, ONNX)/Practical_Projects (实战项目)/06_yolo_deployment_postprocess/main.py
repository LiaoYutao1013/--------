from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


def letterbox_params(original_h: int, original_w: int, target_h: int, target_w: int) -> dict[str, float]:
    ratio = min(target_w / original_w, target_h / original_h)
    resized_w = round(original_w * ratio)
    resized_h = round(original_h * ratio)
    pad_w = target_w - resized_w
    pad_h = target_h - resized_h
    return {
        "ratio": ratio,
        "pad_left": pad_w / 2,
        "pad_top": pad_h / 2,
        "resized_w": resized_w,
        "resized_h": resized_h,
    }


def restore_box(box: Box, params: dict[str, float], original_h: int, original_w: int) -> Box:
    x1 = (box.x1 - params["pad_left"]) / params["ratio"]
    y1 = (box.y1 - params["pad_top"]) / params["ratio"]
    x2 = (box.x2 - params["pad_left"]) / params["ratio"]
    y2 = (box.y2 - params["pad_top"]) / params["ratio"]
    return Box(
        x1=max(0.0, min(original_w, x1)),
        y1=max(0.0, min(original_h, y1)),
        x2=max(0.0, min(original_w, x2)),
        y2=max(0.0, min(original_h, y2)),
        score=box.score,
        class_id=box.class_id,
    )


def area(box: Box) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def iou(left: Box, right: Box) -> float:
    inter_x1 = max(left.x1, right.x1)
    inter_y1 = max(left.y1, right.y1)
    inter_x2 = min(left.x2, right.x2)
    inter_y2 = min(left.y2, right.y2)
    inter = area(Box(inter_x1, inter_y1, inter_x2, inter_y2, 1.0, left.class_id))
    union = area(left) + area(right) - inter
    return 0.0 if union <= 0 else inter / union


def nms(boxes: list[Box], iou_threshold: float) -> list[Box]:
    kept: list[Box] = []
    candidates = sorted(boxes, key=lambda box: box.score, reverse=True)
    while candidates:
        current = candidates.pop(0)
        kept.append(current)
        candidates = [
            box
            for box in candidates
            if box.class_id != current.class_id or iou(current, box) <= iou_threshold
        ]
    return kept


def make_demo_predictions() -> list[Box]:
    return [
        Box(180, 140, 360, 350, 0.92, 0),
        Box(188, 148, 352, 342, 0.86, 0),
        Box(420, 170, 520, 310, 0.78, 1),
        Box(70, 95, 160, 220, 0.21, 0),
        Box(418, 168, 518, 308, 0.64, 1),
    ]


def dynamic_profile(target_size: int) -> dict[str, list[int]]:
    return {
        "min": [1, 3, target_size // 2, target_size // 2],
        "opt": [4, 3, target_size, target_size],
        "max": [8, 3, target_size, target_size],
    }


def calibration_summary(image_shapes: list[tuple[int, int]]) -> dict[str, object]:
    areas = [height * width for height, width in image_shapes]
    aspect_ratios = [round(width / height, 3) for height, width in image_shapes]
    return {
        "sample_count": len(image_shapes),
        "min_area": min(areas),
        "max_area": max(areas),
        "aspect_ratios": aspect_ratios,
        "covers_wide_images": any(ratio > 1.5 for ratio in aspect_ratios),
        "covers_tall_images": any(ratio < 0.75 for ratio in aspect_ratios),
    }


def main() -> int:
    original_h, original_w = 720, 1280
    target_h, target_w = 640, 640
    conf_threshold = 0.25
    iou_threshold = 0.45

    params = letterbox_params(original_h, original_w, target_h, target_w)
    raw_predictions = make_demo_predictions()
    filtered = [box for box in raw_predictions if box.score >= conf_threshold]
    restored = [restore_box(box, params, original_h, original_w) for box in filtered]
    kept = nms(restored, iou_threshold)

    report = {
        "letterbox": params,
        "confidence_threshold": conf_threshold,
        "nms_iou_threshold": iou_threshold,
        "raw_count": len(raw_predictions),
        "after_confidence_filter": len(filtered),
        "after_nms": len(kept),
        "kept_boxes": [asdict(box) for box in kept],
        "tensorrt_dynamic_profile": dynamic_profile(640),
        "int8_calibration": calibration_summary([(720, 1280), (1080, 1920), (1280, 720), (640, 640), (1200, 800)]),
    }

    artifacts = Path(__file__).resolve().parent / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "yolo_postprocess_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("YOLO deployment postprocess report")
    print(f"letterbox ratio: {params['ratio']:.6f}")
    print(f"padding left/top: {params['pad_left']:.1f}/{params['pad_top']:.1f}")
    print(f"raw boxes: {len(raw_predictions)}")
    print(f"after confidence filter: {len(filtered)}")
    print(f"after NMS: {len(kept)}")
    for index, box in enumerate(kept, start=1):
        print(f"box {index}: cls={box.class_id} score={box.score:.2f} xyxy=({box.x1:.1f}, {box.y1:.1f}, {box.x2:.1f}, {box.y2:.1f})")
    print(f"dynamic profile: {report['tensorrt_dynamic_profile']}")
    print("artifact: artifacts/yolo_postprocess_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
