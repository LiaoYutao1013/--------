from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECTS = [
    "01_deployment_pipeline",
    "02_onnx_export_consistency",
    "03_tensorrt_engine_planner",
    "04_quantization_ptq",
    "05_deployment_readiness_checklist",
    "06_yolo_deployment_postprocess",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    failures: list[str] = []

    for project in PROJECTS:
        project_dir = root / project
        print(f"\n=== Running {project} ===", flush=True)
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=project_dir,
            text=True,
        )
        if completed.returncode != 0:
            failures.append(project)

    if failures:
        print("\nFailed projects:")
        for project in failures:
            print(f"- {project}")
        return 1

    print("\nAll deployment practice projects ran successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
