from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KNOWN_EXPERIMENTS = [
    "protocol_v1_curvevel_indomain",
    "protocol_v1_flatvel_to_curvevel",
    "protocol_v1_flatfault_indomain",
    "protocol_v1_flat_curve_to_flatfault",
    "protocol_v1_flatvel_indomain",
    "protocol_v1_curvevel_to_flatvel",
    "matrix_flatvel_a_subset500",
    "matrix_curvevel_a_subset500",
    "matrix_flatfault_a_subset500",
    "matrix_flatvel_to_curvevel_subset500",
    "matrix_curvevel_to_flatvel_subset500",
    "matrix_flat_curve_to_flatfault_subset500",
    "real_dinov2_curvevel_frozen_1ep_smoke",
]

MODEL_DIR_NAMES = ("torch_cnn_baseline", "dummy_dinov2_frozen", "dummy_dinov2_lora")


def _stringify_paths(paths: list[Path]) -> list[str]:
    return [str(path) for path in sorted(paths)]


def _collect_status(item: dict[str, Any]) -> str:
    if not Path(item["experiment_dir"]).exists():
        return "missing"
    missing_files = set(item["missing_files"])
    if not missing_files:
        return "complete"
    if item["history_files"] == []:
        return "missing_history"
    if item["checkpoint_files"] == []:
        return "missing_checkpoint"
    if item["test_metrics_csv"] == []:
        return "missing_metrics"
    if item["prediction_examples_dir"] == []:
        return "missing_prediction_examples"
    return "partial"


def _scan_experiment(experiment_dir: Path, experiment_name: str) -> dict[str, Any]:
    model_dirs = [path for path in (experiment_dir / name for name in MODEL_DIR_NAMES) if path.exists()]
    history_files = list(experiment_dir.rglob("*training_history.csv"))
    checkpoint_files = list(experiment_dir.rglob("*.pt")) + list(experiment_dir.rglob("*.pth"))
    comparison_json = experiment_dir / "comparison" / "comparison_summary.json"
    comparison_csv = experiment_dir / "comparison" / "comparison_summary.csv"
    prediction_dirs = [path for path in experiment_dir.rglob("prediction_examples") if path.is_dir()]
    metrics_csv = [path for path in experiment_dir.rglob("test_metrics.csv")]
    loss_png = [path for path in experiment_dir.rglob("*loss*.png")]
    report_md = [path for path in experiment_dir.rglob("*.md")]
    missing_files: list[str] = []
    if not comparison_json.exists():
        missing_files.append("comparison_summary.json")
    if not comparison_csv.exists():
        missing_files.append("comparison_summary.csv")
    if not history_files:
        missing_files.append("training_history.csv")
    if not checkpoint_files:
        missing_files.append("checkpoint")
    if not metrics_csv:
        missing_files.append("test_metrics.csv")
    if not prediction_dirs:
        missing_files.append("prediction_examples")
    item = {
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "model_dirs": _stringify_paths(model_dirs),
        "history_files": _stringify_paths(history_files),
        "checkpoint_files": _stringify_paths(checkpoint_files),
        "comparison_summary_json": str(comparison_json) if comparison_json.exists() else "",
        "comparison_summary_csv": str(comparison_csv) if comparison_csv.exists() else "",
        "prediction_examples_dir": _stringify_paths(prediction_dirs),
        "test_metrics_csv": _stringify_paths(metrics_csv),
        "loss_png_files": _stringify_paths(loss_png),
        "report_md_files": _stringify_paths(report_md),
        "status": "",
        "missing_files": missing_files,
    }
    item["status"] = _collect_status(item)
    return item


def build_results_index(outputs_root: str | Path) -> dict[str, Any]:
    root = Path(outputs_root)
    experiments = [_scan_experiment(root / name, name) for name in KNOWN_EXPERIMENTS]
    return {"outputs_root": str(root), "experiments": experiments, "count": len(experiments)}


def write_results_index_outputs(payload: dict[str, Any], output_json: str | Path, output_md: str | Path) -> None:
    json_path = Path(output_json)
    md_path = Path(output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Results Index",
        "",
        f"- `outputs_root`: `{payload.get('outputs_root', '')}`",
        f"- `count`: `{payload.get('count', 0)}`",
        "",
        "| experiment_name | status | model_dirs | histories | checkpoints | metrics_csv | reports | missing_files |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in payload.get("experiments", []):
        lines.append(
            f"| {item['experiment_name']} | {item['status']} | {len(item['model_dirs'])} | {len(item['history_files'])} | "
            f"{len(item['checkpoint_files'])} | {len(item['test_metrics_csv'])} | {len(item['report_md_files'])} | "
            f"{', '.join(item['missing_files'])} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描 outputs 目录并生成统一结果索引。")
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_results_index(args.outputs_root)
    write_results_index_outputs(payload, args.output_json, args.output_md)
    print(f"写出结果索引: {args.output_json}")
    print(f"实验数量: {payload['count']}")


if __name__ == "__main__":
    main()
