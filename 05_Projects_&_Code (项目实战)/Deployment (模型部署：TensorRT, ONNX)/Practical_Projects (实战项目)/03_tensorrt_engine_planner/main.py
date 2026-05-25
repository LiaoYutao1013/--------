from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Layer:
    name: str
    op_type: str
    output_shape: tuple[int, int, int]
    fp32_ms: float


@dataclass
class EnginePlan:
    precision: str
    min_shape: tuple[int, int, int, int]
    opt_shape: tuple[int, int, int, int]
    max_shape: tuple[int, int, int, int]
    workspace_mib: int
    fused_patterns: list[str]
    estimated_latency_ms: float
    trtexec_command: str


LAYERS = [
    Layer("stem_conv", "Conv", (32, 320, 320), 1.20),
    Layer("stem_bn", "BatchNormalization", (32, 320, 320), 0.32),
    Layer("stem_relu", "Relu", (32, 320, 320), 0.18),
    Layer("block1_conv", "Conv", (64, 160, 160), 1.05),
    Layer("block1_bn", "BatchNormalization", (64, 160, 160), 0.22),
    Layer("block1_relu", "Relu", (64, 160, 160), 0.12),
    Layer("head_conv", "Conv", (255, 80, 80), 0.85),
    Layer("decode", "CustomDecode", (25200, 85, 1), 0.40),
]


def find_fusion_patterns(layers: list[Layer]) -> list[str]:
    patterns: list[str] = []
    index = 0
    while index <= len(layers) - 3:
        current = layers[index : index + 3]
        if [layer.op_type for layer in current] == ["Conv", "BatchNormalization", "Relu"]:
            patterns.append("+".join(layer.name for layer in current))
            index += 3
        else:
            index += 1
    return patterns


def estimate_latency(layers: list[Layer], precision: str, fused_count: int) -> float:
    base = sum(layer.fp32_ms for layer in layers)
    precision_factor = {"FP32": 1.0, "FP16": 0.62, "INT8": 0.45}[precision]
    fusion_saving = 0.18 * fused_count
    custom_plugin_penalty = 0.2 if any(layer.op_type.startswith("Custom") for layer in layers) else 0.0
    return round(max(0.1, base * precision_factor - fusion_saving + custom_plugin_penalty), 3)


def build_trtexec_command(precision: str, workspace_mib: int, min_shape: tuple[int, int, int, int], opt_shape: tuple[int, int, int, int], max_shape: tuple[int, int, int, int]) -> str:
    flags = [
        "trtexec",
        "--onnx=model.onnx",
        "--saveEngine=model.engine",
        f"--memPoolSize=workspace:{workspace_mib}",
        f"--minShapes=images:{'x'.join(map(str, min_shape))}",
        f"--optShapes=images:{'x'.join(map(str, opt_shape))}",
        f"--maxShapes=images:{'x'.join(map(str, max_shape))}",
    ]
    if precision == "FP16":
        flags.append("--fp16")
    elif precision == "INT8":
        flags.extend(["--int8", "--calib=calibration.cache"])
    return " ".join(flags)


def make_plan(precision: str, image_size: int, max_batch: int, workspace_mib: int) -> EnginePlan:
    min_shape = (1, 3, image_size // 2, image_size // 2)
    opt_shape = (max(1, max_batch // 2), 3, image_size, image_size)
    max_shape = (max_batch, 3, image_size, image_size)
    fused_patterns = find_fusion_patterns(LAYERS)
    command = build_trtexec_command(precision, workspace_mib, min_shape, opt_shape, max_shape)
    return EnginePlan(
        precision=precision,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
        workspace_mib=workspace_mib,
        fused_patterns=fused_patterns,
        estimated_latency_ms=estimate_latency(LAYERS, precision, len(fused_patterns)),
        trtexec_command=command,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a TensorRT engine profile and trtexec command.")
    parser.add_argument("--precision", choices=["FP32", "FP16", "INT8"], default="FP16")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--workspace-mib", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.image_size <= 0 or args.max_batch <= 0 or args.workspace_mib <= 0:
        raise ValueError("image-size, max-batch, and workspace-mib must be positive")

    plan = make_plan(args.precision, args.image_size, args.max_batch, args.workspace_mib)
    artifacts = Path(__file__).resolve().parent / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "tensorrt_engine_plan.json").write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")
    (artifacts / "trtexec_command.txt").write_text(plan.trtexec_command + "\n", encoding="utf-8")

    print("TensorRT engine plan")
    print(f"precision: {plan.precision}")
    print(f"profile min/opt/max: {plan.min_shape} / {plan.opt_shape} / {plan.max_shape}")
    print(f"fusion patterns: {len(plan.fused_patterns)}")
    for pattern in plan.fused_patterns:
        print(f"- {pattern}")
    print(f"estimated latency: {plan.estimated_latency_ms:.3f} ms")
    print("\ntrtexec command:")
    print(plan.trtexec_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
