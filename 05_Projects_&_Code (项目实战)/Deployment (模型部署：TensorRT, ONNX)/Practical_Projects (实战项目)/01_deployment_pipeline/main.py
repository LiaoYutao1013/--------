from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class StageResult:
    name: str
    status: str
    artifact: str
    checks: list[str]
    elapsed_ms: float


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_stage(name: str, artifact: Path, checks: list[str], payload: dict) -> StageResult:
    started = time.perf_counter()
    write_json(artifact, payload)
    elapsed_ms = (time.perf_counter() - started) * 1000
    project_dir = Path(__file__).resolve().parent
    return StageResult(
        name=name,
        status="PASS",
        artifact=str(artifact.relative_to(project_dir)),
        checks=checks,
        elapsed_ms=round(elapsed_ms, 3),
    )


def build_pipeline(target: str, batch_size: int, input_size: tuple[int, int]) -> list[StageResult]:
    artifacts = Path(__file__).resolve().parent / "artifacts"
    height, width = input_size

    stages = [
        run_stage(
            "checkpoint_validation",
            artifacts / "01_checkpoint.json",
            ["weights_found", "input_signature_recorded", "class_names_recorded"],
            {
                "format": "demo-checkpoint",
                "input": {"shape": [batch_size, 3, height, width], "dtype": "float32"},
                "classes": ["part_ok", "scratch", "missing_part"],
            },
        ),
        run_stage(
            "export_onnx",
            artifacts / "02_model.onnx.json",
            ["opset=17", "dynamic_batch=true", "numerical_parity_target=1e-5"],
            {
                "format": "onnx-placeholder",
                "opset": 17,
                "dynamic_axes": {"images": [0], "outputs": [0]},
                "operators": ["Conv", "BatchNormalization", "Relu", "Gemm"],
            },
        ),
        run_stage(
            "graph_optimization",
            artifacts / "03_optimized_graph.json",
            ["constant_folding", "conv_bn_fusion", "unused_nodes_removed"],
            {
                "removed_nodes": 2,
                "fused_patterns": ["Conv+BatchNormalization"],
                "estimated_latency_ms": 4.8 if target == "server" else 12.5,
            },
        ),
        run_stage(
            "engine_selection",
            artifacts / "04_engine_plan.json",
            ["runtime_selected", "device_constraints_checked"],
            {
                "runtime": "TensorRT" if target == "server" else "ONNX Runtime Mobile",
                "precision": "FP16" if target == "server" else "INT8",
                "target": target,
            },
        ),
        run_stage(
            "quantization_plan",
            artifacts / "05_quantization_plan.json",
            ["calibration_policy_defined", "fallback_precision_defined"],
            {
                "method": "PTQ",
                "calibration_samples": 256,
                "fallback_layers": ["first_conv", "output_head"],
            },
        ),
        run_stage(
            "deployment_package",
            artifacts / "06_package_manifest.json",
            ["preprocess_version_pinned", "postprocess_version_pinned", "smoke_test_required"],
            {
                "files": [
                    "model.engine" if target == "server" else "model.ort",
                    "preprocess.py",
                    "postprocess.py",
                    "config.json",
                ],
                "healthcheck": "/healthz" if target == "server" else "local_smoke_test",
            },
        ),
    ]

    write_json(
        artifacts / "pipeline_manifest.json",
        {
            "target": target,
            "batch_size": batch_size,
            "input_size": [height, width],
            "stages": [asdict(stage) for stage in stages],
        },
    )
    return stages


def print_table(stages: list[StageResult]) -> None:
    print("Deployment pipeline result")
    print("-" * 88)
    print(f"{'stage':28} {'status':8} {'elapsed_ms':>10} artifact")
    print("-" * 88)
    for stage in stages:
        print(f"{stage.name:28} {stage.status:8} {stage.elapsed_ms:10.3f} {stage.artifact}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a model deployment pipeline simulation.")
    parser.add_argument("--target", choices=["server", "edge"], default="server")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=[640, 640])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.input_size[0] <= 0 or args.input_size[1] <= 0:
        raise ValueError("input-size values must be positive")

    stages = build_pipeline(args.target, args.batch_size, tuple(args.input_size))
    print_table(stages)
    print("\nManifest written to artifacts/pipeline_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
