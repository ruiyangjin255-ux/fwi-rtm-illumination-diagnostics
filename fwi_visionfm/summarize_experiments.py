from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _metric(data: dict[str, object], key: str) -> object:
    value = data.get(key, "N/A")
    return "N/A" if value is None else value


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_experiments(outputs_root: str | Path) -> dict[str, str]:
    root = Path(outputs_root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []

    torch_cpu_metrics = _read_json_if_exists(root / "torch_cpu_experiment" / "metrics.json")
    torch_cpu_summary = _read_json_if_exists(root / "torch_cpu_experiment" / "experiment_summary.json")
    if torch_cpu_metrics is not None and torch_cpu_summary is not None:
        rows.append(
            {
                "experiment_name": "torch_cpu_experiment",
                "data_type": str(torch_cpu_summary.get("data_type", "synthetic")),
                "bridge": str(torch_cpu_summary.get("bridge", "simple")),
                "aggregation": str(torch_cpu_summary.get("aggregation", "source_attention")),
                "decoder": str(torch_cpu_summary.get("decoder", "bounded")),
                "test_mae": _metric(torch_cpu_metrics, "test_mae"),
                "test_rmse": _metric(torch_cpu_metrics, "test_rmse"),
                "test_relative_mae": _metric(torch_cpu_metrics, "test_relative_mae"),
                "test_relative_rmse": _metric(torch_cpu_metrics, "test_relative_rmse"),
                "test_psnr": _metric(torch_cpu_metrics, "test_psnr"),
                "test_ssim": _metric(torch_cpu_metrics, "test_ssim"),
                "test_gradient_error": _metric(torch_cpu_metrics, "test_gradient_error"),
                "notes": "CPU synthetic baseline",
            }
        )

    ablation_best = _read_json_if_exists(root / "torch_ablation" / "best_config.json")
    if ablation_best is not None:
        rows.append(
            {
                "experiment_name": "torch_ablation_best",
                "data_type": "synthetic",
                "bridge": str(ablation_best.get("bridge", "unknown")),
                "aggregation": str(ablation_best.get("aggregation", "unknown")),
                "decoder": str(ablation_best.get("decoder", "unknown")),
                "test_mae": _metric(ablation_best, "test_mae"),
                "test_rmse": _metric(ablation_best, "test_rmse"),
                "test_relative_mae": _metric(ablation_best, "test_relative_mae"),
                "test_relative_rmse": _metric(ablation_best, "test_relative_rmse"),
                "test_psnr": _metric(ablation_best, "test_psnr"),
                "test_ssim": _metric(ablation_best, "test_ssim"),
                "test_gradient_error": _metric(ablation_best, "test_gradient_error"),
                "notes": "Best smoke-scale ablation config",
            }
        )

    openfwi_metrics_path = root / "openfwi_small_experiment" / "metrics.json"
    openfwi_summary_path = root / "openfwi_small_experiment" / "experiment_summary.json"
    if openfwi_metrics_path.exists() and openfwi_summary_path.exists():
        openfwi_metrics = json.loads(openfwi_metrics_path.read_text(encoding="utf-8"))
        openfwi_summary = json.loads(openfwi_summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment_name": "openfwi_small_experiment",
                "data_type": str(openfwi_summary.get("data_type", "openfwi_small")),
                "bridge": str(openfwi_summary.get("bridge", "channel_stack")),
                "aggregation": str(openfwi_summary.get("aggregation", "max")),
                "decoder": str(openfwi_summary.get("decoder", "bounded")),
                "test_mae": _metric(openfwi_metrics, "test_mae"),
                "test_rmse": _metric(openfwi_metrics, "test_rmse"),
                "test_relative_mae": _metric(openfwi_metrics, "test_relative_mae"),
                "test_relative_rmse": _metric(openfwi_metrics, "test_relative_rmse"),
                "test_psnr": _metric(openfwi_metrics, "test_psnr"),
                "test_ssim": _metric(openfwi_metrics, "test_ssim"),
                "test_gradient_error": _metric(openfwi_metrics, "test_gradient_error"),
                "notes": "CPU OpenFWI-style small-scale experiment",
            }
        )

    openfwi_scale_best_path = root / "openfwi_scale_study" / "best_scale_config.json"
    if openfwi_scale_best_path.exists():
        openfwi_scale_best = json.loads(openfwi_scale_best_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment_name": "openfwi_scale_study_best",
                "data_type": "openfwi_scale_study",
                "bridge": str(openfwi_scale_best.get("bridge", "channel_stack")),
                "aggregation": str(openfwi_scale_best.get("aggregation", "max")),
                "decoder": str(openfwi_scale_best.get("decoder", "bounded")),
                "test_mae": _metric(openfwi_scale_best, "test_mae"),
                "test_rmse": _metric(openfwi_scale_best, "test_rmse"),
                "test_relative_mae": _metric(openfwi_scale_best, "test_relative_mae"),
                "test_relative_rmse": _metric(openfwi_scale_best, "test_relative_rmse"),
                "test_psnr": _metric(openfwi_scale_best, "test_psnr"),
                "test_ssim": _metric(openfwi_scale_best, "test_ssim"),
                "test_gradient_error": _metric(openfwi_scale_best, "test_gradient_error"),
                "notes": "Best CPU OpenFWI scale-study result",
            }
        )

    csv_path = root / "experiment_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_name",
                "data_type",
                "bridge",
                "aggregation",
                "decoder",
                "test_mae",
                "test_rmse",
                "test_relative_mae",
                "test_relative_rmse",
                "test_psnr",
                "test_ssim",
                "test_gradient_error",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Experiment Comparison",
        "",
        "| experiment_name | data_type | bridge | aggregation | decoder | test_mae | test_rmse | test_relative_mae | test_relative_rmse | test_psnr | test_ssim | test_gradient_error | notes |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['experiment_name']} | {row['data_type']} | {row['bridge']} | {row['aggregation']} | {row['decoder']} | "
            f"{row['test_mae']} | {row['test_rmse']} | {row['test_relative_mae']} | {row['test_relative_rmse']} | "
            f"{row['test_psnr']} | {row['test_ssim']} | {row['test_gradient_error']} | {row['notes']} |"
        )
    md_path = root / "experiment_comparison.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return {"csv": str(csv_path), "md": str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize torch CPU, ablation, and OpenFWI small experiment results.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    outputs = summarize_experiments(args.outputs_root)
    print(f"Wrote comparison csv to {outputs['csv']}")
    print(f"Wrote comparison md to {outputs['md']}")


if __name__ == "__main__":
    main()
