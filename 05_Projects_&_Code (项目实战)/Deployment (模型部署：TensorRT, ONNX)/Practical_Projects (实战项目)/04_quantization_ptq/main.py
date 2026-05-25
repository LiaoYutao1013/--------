from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


WEIGHTS = [
    [0.25, -0.18, 0.33, 0.47],
    [-0.42, 0.55, 0.12, -0.08],
    [0.31, 0.27, -0.38, 0.19],
]
BIAS = [0.03, -0.02, 0.01]


def dot(row: list[float], vector: list[float]) -> float:
    return sum(weight * value for weight, value in zip(row, vector))


def fp32_infer(sample: list[float]) -> list[float]:
    return [max(0.0, dot(row, sample) + bias) for row, bias in zip(WEIGHTS, BIAS)]


def generate_samples(count: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(4)] for _ in range(count)]


def symmetric_scale(max_abs: float) -> float:
    return max(max_abs, 1e-12) / 127.0


def quantize_value(value: float, scale: float) -> int:
    return max(-127, min(127, round(value / scale)))


def dequantize_value(value: int, scale: float) -> float:
    return value * scale


def quantize_matrix(matrix: list[list[float]], scale: float) -> list[list[int]]:
    return [[quantize_value(value, scale) for value in row] for row in matrix]


def int8_infer(sample: list[float], input_scale: float, weight_scale: float, output_scale: float, q_weights: list[list[int]]) -> list[float]:
    q_input = [quantize_value(value, input_scale) for value in sample]
    outputs: list[float] = []
    accumulator_scale = input_scale * weight_scale
    for row, bias in zip(q_weights, BIAS):
        accumulator = sum(weight * value for weight, value in zip(row, q_input))
        fp_value = accumulator * accumulator_scale + bias
        activated = max(0.0, fp_value)
        q_output = quantize_value(activated, output_scale)
        outputs.append(dequantize_value(q_output, output_scale))
    return outputs


def max_abs(values: list[float]) -> float:
    return max(abs(value) for value in values)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small INT8 PTQ quantization experiment.")
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.calibration_samples <= 0 or args.test_samples <= 0:
        raise ValueError("sample counts must be positive")

    calibration = generate_samples(args.calibration_samples, seed=7)
    tests = generate_samples(args.test_samples, seed=19)

    input_scale = symmetric_scale(max(max_abs(sample) for sample in calibration))
    weight_scale = symmetric_scale(max(max_abs(row) for row in WEIGHTS))
    activation_max = max(max_abs(fp32_infer(sample)) for sample in calibration)
    output_scale = symmetric_scale(activation_max)
    q_weights = quantize_matrix(WEIGHTS, weight_scale)

    sample_errors = []
    for sample in tests:
        fp32_output = fp32_infer(sample)
        int8_output = int8_infer(sample, input_scale, weight_scale, output_scale, q_weights)
        sample_errors.append(max(abs(a - b) for a, b in zip(fp32_output, int8_output)))

    report = {
        "calibration_samples": args.calibration_samples,
        "test_samples": args.test_samples,
        "input_scale": input_scale,
        "weight_scale": weight_scale,
        "output_scale": output_scale,
        "max_abs_error": max(sample_errors),
        "mean_abs_error": mean(sample_errors),
        "q_weights": q_weights,
    }

    artifacts = Path(__file__).resolve().parent / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "quantization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("INT8 PTQ quantization report")
    print(f"input scale: {input_scale:.8f}")
    print(f"weight scale: {weight_scale:.8f}")
    print(f"output scale: {output_scale:.8f}")
    print(f"max abs error: {report['max_abs_error']:.8f}")
    print(f"mean abs error: {report['mean_abs_error']:.8f}")
    print("artifact: artifacts/quantization_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
