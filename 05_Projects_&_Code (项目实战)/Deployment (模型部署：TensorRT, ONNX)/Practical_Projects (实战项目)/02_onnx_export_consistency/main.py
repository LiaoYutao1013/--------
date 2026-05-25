from __future__ import annotations

import json
import math
from pathlib import Path


SUPPORTED_OPS = {"Gemm", "Relu", "Softmax"}


def matmul_vector(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(weight * value for weight, value in zip(row, vector)) for row in matrix]


def add_bias(values: list[float], bias: list[float]) -> list[float]:
    return [value + bias_value for value, bias_value in zip(values, bias)]


def relu(values: list[float]) -> list[float]:
    return [max(0.0, value) for value in values]


def softmax(values: list[float]) -> list[float]:
    offset = max(values)
    exps = [math.exp(value - offset) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def source_model(sample: list[float], params: dict) -> list[float]:
    hidden = relu(add_bias(matmul_vector(params["linear1_weight"], sample), params["linear1_bias"]))
    logits = add_bias(matmul_vector(params["linear2_weight"], hidden), params["linear2_bias"])
    return softmax(logits)


def exported_graph_model(sample: list[float], graph: dict) -> list[float]:
    tensors: dict[str, list[float]] = {"images": sample}
    initializers = graph["initializers"]

    for node in graph["nodes"]:
        op_type = node["op_type"]
        if op_type == "Gemm":
            input_name, weight_name, bias_name = node["inputs"]
            output_name = node["outputs"][0]
            tensors[output_name] = add_bias(
                matmul_vector(initializers[weight_name], tensors[input_name]),
                initializers[bias_name],
            )
        elif op_type == "Relu":
            tensors[node["outputs"][0]] = relu(tensors[node["inputs"][0]])
        elif op_type == "Softmax":
            tensors[node["outputs"][0]] = softmax(tensors[node["inputs"][0]])
        else:
            raise ValueError(f"Unsupported op in exported graph: {op_type}")

    return tensors[graph["outputs"][0]["name"]]


def export_graph(params: dict) -> dict:
    return {
        "ir_version": "demo-onnx-json",
        "opset": 17,
        "inputs": [{"name": "images", "shape": ["batch", 4], "dtype": "float32"}],
        "outputs": [{"name": "probabilities", "shape": ["batch", 2], "dtype": "float32"}],
        "dynamic_axes": {"images": [0], "probabilities": [0]},
        "nodes": [
            {"name": "linear1", "op_type": "Gemm", "inputs": ["images", "linear1_weight", "linear1_bias"], "outputs": ["hidden"]},
            {"name": "relu1", "op_type": "Relu", "inputs": ["hidden"], "outputs": ["hidden_relu"]},
            {"name": "linear2", "op_type": "Gemm", "inputs": ["hidden_relu", "linear2_weight", "linear2_bias"], "outputs": ["logits"]},
            {"name": "prob", "op_type": "Softmax", "inputs": ["logits"], "outputs": ["probabilities"]},
        ],
        "initializers": params,
    }


def validate_graph(graph: dict) -> list[str]:
    errors: list[str] = []
    if graph.get("opset", 0) < 13:
        errors.append("opset must be >= 13 for this deployment target")

    for node in graph["nodes"]:
        if node["op_type"] not in SUPPORTED_OPS:
            errors.append(f"unsupported operator: {node['op_type']}")

    if "images" not in graph.get("dynamic_axes", {}):
        errors.append("input dynamic batch axis is missing")

    return errors


def max_abs_error(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    artifacts = project_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)

    params = {
        "linear1_weight": [
            [0.20, -0.10, 0.40, 0.10],
            [-0.30, 0.50, 0.10, -0.20],
            [0.60, 0.10, -0.30, 0.20],
        ],
        "linear1_bias": [0.01, -0.02, 0.03],
        "linear2_weight": [
            [0.30, -0.20, 0.50],
            [-0.40, 0.60, 0.20],
        ],
        "linear2_bias": [0.04, -0.01],
    }
    samples = [
        [0.10, 0.40, -0.20, 0.70],
        [0.90, -0.10, 0.30, 0.20],
        [-0.50, 0.80, 0.10, -0.30],
    ]

    graph = export_graph(params)
    graph_path = artifacts / "model.onnx.json"
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")

    errors = validate_graph(graph)
    if errors:
        print("Graph validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    tolerance = 1e-9
    report = []
    for index, sample in enumerate(samples):
        expected = source_model(sample, params)
        actual = exported_graph_model(sample, graph)
        error = max_abs_error(expected, actual)
        report.append({"sample": index, "expected": expected, "actual": actual, "max_abs_error": error})

    report_path = artifacts / "parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    worst_error = max(item["max_abs_error"] for item in report)
    print("ONNX-style export consistency report")
    print(f"opset: {graph['opset']}")
    print(f"supported ops: {', '.join(sorted(SUPPORTED_OPS))}")
    print(f"dynamic axes: {graph['dynamic_axes']}")
    print(f"worst max abs error: {worst_error:.12f}")
    print(f"status: {'PASS' if worst_error <= tolerance else 'FAIL'}")
    print("artifacts: artifacts/model.onnx.json, artifacts/parity_report.json")
    return 0 if worst_error <= tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
