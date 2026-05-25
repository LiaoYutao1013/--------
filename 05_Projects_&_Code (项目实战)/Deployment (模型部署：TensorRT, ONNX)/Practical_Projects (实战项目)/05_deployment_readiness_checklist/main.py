from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def check_preprocess(manifest: dict) -> Check:
    preprocess = manifest.get("preprocess", {})
    passed = preprocess.get("train_version") == preprocess.get("deploy_version")
    return Check(
        "preprocess_consistency",
        passed,
        f"train={preprocess.get('train_version')} deploy={preprocess.get('deploy_version')}",
    )


def check_postprocess(manifest: dict) -> Check:
    postprocess = manifest.get("postprocess", {})
    passed = postprocess.get("validation_version") == postprocess.get("deploy_version")
    return Check(
        "postprocess_consistency",
        passed,
        f"validation={postprocess.get('validation_version')} deploy={postprocess.get('deploy_version')}",
    )


def check_parity(manifest: dict) -> Check:
    parity = manifest.get("parity", {})
    error = parity.get("onnx_max_abs_error", float("inf"))
    limit = parity.get("allowed_max_abs_error", 0.0)
    return Check("onnx_parity", error <= limit, f"error={error} limit={limit}")


def check_dynamic_inputs(manifest: dict) -> Check:
    tests = manifest.get("dynamic_input_tests", [])
    passed = bool(tests) and all(test.get("passed") for test in tests)
    shapes = [f"b{test.get('batch')}:{test.get('height')}x{test.get('width')}" for test in tests]
    return Check("dynamic_input_coverage", passed, ", ".join(shapes))


def check_performance(manifest: dict) -> Check:
    performance = manifest.get("performance", {})
    required = ["latency_p50_ms", "latency_p95_ms", "throughput_fps", "gpu_memory_mib"]
    missing = [name for name in required if name not in performance]
    return Check("performance_recorded", not missing, "missing=" + ",".join(missing) if missing else "all metrics recorded")


def check_error_handling(manifest: dict) -> Check:
    handling = manifest.get("error_handling", {})
    required = ["empty_image", "wrong_dtype", "oversized_image", "corrupt_input"]
    missing = [name for name in required if not handling.get(name)]
    return Check("error_handling", not missing, "missing=" + ",".join(missing) if missing else "all required cases covered")


def run_checks(manifest: dict) -> list[Check]:
    return [
        check_preprocess(manifest),
        check_postprocess(manifest),
        check_parity(manifest),
        check_dynamic_inputs(manifest),
        check_performance(manifest),
        check_error_handling(manifest),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a deployment manifest against a readiness checklist.")
    parser.add_argument("--manifest", default="deployment_manifest.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = project_dir / manifest_path

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = run_checks(manifest)
    passed = all(check.passed for check in checks)

    report = {
        "model_name": manifest.get("model_name", "unknown"),
        "passed": passed,
        "checks": [asdict(check) for check in checks],
    }

    artifacts = project_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "readiness_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Deployment readiness audit")
    print(f"model: {report['model_name']}")
    print(f"overall: {'PASS' if passed else 'FAIL'}")
    for check in checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"{mark:4} {check.name:26} {check.detail}")
    print("artifact: artifacts/readiness_report.json")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
