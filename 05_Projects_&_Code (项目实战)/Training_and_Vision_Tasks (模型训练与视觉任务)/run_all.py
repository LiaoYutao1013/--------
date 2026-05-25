from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


TASKS = [
    "01_from_scratch_classifier",
    "02_minibatch_training",
    "03_frozen_feature_transfer",
    "04_finetune_last_layers",
    "05_augmentation_ablation",
    "06_hyperparameter_search",
    "07_class_imbalance_training",
    "08_cross_validation",
    "09_object_detection_baseline",
    "10_segmentation_baseline",
    "11_contrastive_pretraining",
    "12_training_report_pack"
]


def main() -> int:
    root = Path(__file__).resolve().parent
    failures: list[str] = []
    for task in TASKS:
        print(f"=== Running {task} ===", flush=True)
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
            print(f"- {task}")
        return 1
    print(f"All {len(TASKS)} training and vision tasks ran successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
