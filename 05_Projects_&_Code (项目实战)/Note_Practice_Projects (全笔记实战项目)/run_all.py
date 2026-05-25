from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROJECTS = [
    "001_00_cv_master_hub_concept_graph_and_md_9354d11f",
    "002_00_cv_master_hub_cv_learning_roadmap_md_5d192589",
    "003_00_cv_master_hub_cv_md_793a26b9",
    "004_00_cv_master_hub_md_e6c896d0",
    "005_00_cv_master_hub_md_1549742b",
    "006_00_cv_master_hub_md_140deef7",
    "007_01_foundations_digital_image_processing_op_c1dc7758",
    "008_01_foundations_digital_image_processing_op_64d1be76",
    "009_01_foundations_digital_image_processing_op_a6f7138f",
    "010_01_foundations_digital_image_processing_op_8b2338fe",
    "011_01_foundations_digital_image_processing_op_7dedbd2b",
    "012_01_foundations_digital_image_processing_op_dc3a18e7",
    "013_01_foundations_digital_image_processing_op_01a16c06",
    "014_01_foundations_digital_image_processing_op_061c8e94",
    "015_01_foundations_digital_image_processing_op_520d4e4e",
    "016_01_foundations_digital_image_processing_op_ae36a9a8",
    "017_01_foundations_machine_learning_basic_svm__5b908a4c",
    "018_01_foundations_math_for_cv_md_d7c979e9",
    "019_01_foundations_math_for_cv_md_0f527251",
    "020_01_foundations_math_for_cv_md_9589d328",
    "021_01_foundations_md_3ec69428",
    "022_02_deep_learning_core_architectures_cnn_se_44ef9b12",
    "023_02_deep_learning_core_architectures_cnn_se_42f7c878",
    "024_02_deep_learning_core_architectures_mlp_ot_2d0924b4",
    "025_02_deep_learning_core_architectures_transf_1a657318",
    "026_02_deep_learning_core_architectures_transf_1766aaab",
    "027_02_deep_learning_core_architectures_md_7eb09342",
    "028_02_deep_learning_core_training_techniques__f9457687",
    "029_02_deep_learning_core_training_techniques__40b40571",
    "030_02_deep_learning_core_md_a74f7032",
    "031_03_computer_vision_tasks_3d_vision_nerf_3d_89b72328",
    "032_03_computer_vision_tasks_3d_vision_nerf_3d_a2b92fef",
    "033_03_computer_vision_tasks_3d_vision_nerf_3d_16660329",
    "034_03_computer_vision_tasks_3d_vision_nerf_3d_d15c8794",
    "035_03_computer_vision_tasks_3d_vision_nerf_3d_13c2c5b0",
    "036_03_computer_vision_tasks_3d_vision_nerf_3d_0b78c1eb",
    "037_03_computer_vision_tasks_3d_vision_nerf_3d_9072fa73",
    "038_03_computer_vision_tasks_3d_vision_nerf_3d_98c10382",
    "039_03_computer_vision_tasks_classification_md_7e27e442",
    "040_03_computer_vision_tasks_cv_md_43fed8ff",
    "041_03_computer_vision_tasks_generative_models_bf82d304",
    "042_03_computer_vision_tasks_object_detection__61951ddc",
    "043_03_computer_vision_tasks_object_detection__88359afd",
    "044_03_computer_vision_tasks_object_detection__99ddb4f5",
    "045_03_computer_vision_tasks_object_detection__01bbe503",
    "046_03_computer_vision_tasks_object_detection__6cc0dc59",
    "047_03_computer_vision_tasks_object_detection__43910478",
    "048_03_computer_vision_tasks_object_detection__943521da",
    "049_03_computer_vision_tasks_object_detection__ca3433fa",
    "050_03_computer_vision_tasks_object_detection__8ba83f3f",
    "051_03_computer_vision_tasks_object_detection__284b91b5",
    "052_03_computer_vision_tasks_object_detection__6e1faf69",
    "053_03_computer_vision_tasks_object_detection__1781be87",
    "054_03_computer_vision_tasks_object_detection__38f340f7",
    "055_03_computer_vision_tasks_object_detection__d2bef0f7",
    "056_03_computer_vision_tasks_object_detection__f2f38931",
    "057_03_computer_vision_tasks_segmentation_md_ab3a227e",
    "058_04_papers_literature_2024_readings_2024_md_8ecf15bd",
    "059_04_papers_literature_2025_readings_2025_md_4de9d1cb",
    "060_04_papers_literature_md_b5349313",
    "061_05_projects_code_competition_kaggle_md_09116204",
    "062_05_projects_code_deployment_tensorrt_onnx__3705688c",
    "063_05_projects_code_open_source_labs_md_480e563c",
    "064_05_projects_code_md_e162fbbf",
    "065_99_attachments_pdf_md_19ce69df"
]


def main() -> int:
    root = Path(__file__).resolve().parent
    failures: list[str] = []
    for project in PROJECTS:
        print(f"=== Running {project} ===", flush=True)
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
            print(f"- {project}")
        return 1
    print(f"All {len(PROJECTS)} note practice projects ran successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
