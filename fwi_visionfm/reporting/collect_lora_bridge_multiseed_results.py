from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_parameter_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    trainable_match = re.search(r"trainable parameters:\s*(\d+)", text)
    ratio_match = re.search(r"trainable ratio:\s*([0-9.]+)", text)
    return {
        "trainable_params": int(trainable_match.group(1)) if trainable_match else "NA",
        "trainable_ratio": float(ratio_match.group(1)) if ratio_match else "NA",
    }


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    return float(value)


def _format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def _metric_mean_std(rows: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    values = [_safe_float(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return mean(clean), pstdev(clean)


def _parse_seed(name: str) -> int:
    match = re.match(r".+_seed(\d+)$", name)
    if not match:
        raise ValueError(f"Experiment directory name must match *_seed<seed>, got: {name}")
    return int(match.group(1))


def _collect_one(experiment_dir: Path) -> dict[str, Any]:
    config_path = experiment_dir / "config_resolved.json"
    if not config_path.exists():
        config_path = experiment_dir / "resolved_foundation_config.json"
    config = _read_json(config_path)
    metrics_in = _read_json(experiment_dir / "metrics_in_family_extended.json")
    metrics_cross = _read_json(experiment_dir / "metrics_cross_family_extended.json")
    params = (
        _parse_parameter_report(experiment_dir / "parameter_report.txt")
        if (experiment_dir / "parameter_report.txt").exists()
        else {"trainable_params": "NA", "trainable_ratio": "NA"}
    )
    row = {
        "experiment": experiment_dir.name,
        "seed": _parse_seed(experiment_dir.name),
        "bridge_feature_mode": config.get("bridge_feature_mode", "NA"),
        "in_family_mae": metrics_in.get("mae", "NA"),
        "in_family_rmse": metrics_in.get("rmse", "NA"),
        "in_family_psnr": metrics_in.get("psnr", "NA"),
        "in_family_ssim": metrics_in.get("ssim", "NA"),
        "in_family_edge_mae": metrics_in.get("edge_mae", "NA"),
        "in_family_laplacian_mae": metrics_in.get("laplacian_mae", "NA"),
        "in_family_horizon_gradient_mae": metrics_in.get("horizon_gradient_mae", "NA"),
        "in_family_vertical_gradient_mae": metrics_in.get("vertical_gradient_mae", "NA"),
        "cross_family_mae": metrics_cross.get("mae", "NA"),
        "cross_family_rmse": metrics_cross.get("rmse", "NA"),
        "cross_family_psnr": metrics_cross.get("psnr", "NA"),
        "cross_family_ssim": metrics_cross.get("ssim", "NA"),
        "cross_family_edge_mae": metrics_cross.get("edge_mae", "NA"),
        "cross_family_laplacian_mae": metrics_cross.get("laplacian_mae", "NA"),
        "cross_family_horizon_gradient_mae": metrics_cross.get("horizon_gradient_mae", "NA"),
        "cross_family_vertical_gradient_mae": metrics_cross.get("vertical_gradient_mae", "NA"),
        "generalization_gap_mae": float(metrics_cross["mae"]) - float(metrics_in["mae"]),
        "generalization_gap_rmse": float(metrics_cross["rmse"]) - float(metrics_in["rmse"]),
        "trainable_params": params.get("trainable_params", "NA"),
        "trainable_ratio": params.get("trainable_ratio", "NA"),
        "output_dir": str(experiment_dir),
    }
    return row


def _pick_experiment_dirs(root_dir: Path) -> list[Path]:
    return sorted(
        path for path in root_dir.iterdir()
        if path.is_dir()
        and re.match(r"lora_raw_(repeat3|spectrogram)_seed\d+$", path.name)
        and (path / "metrics_in_family_extended.json").exists()
        and (path / "metrics_cross_family_extended.json").exists()
    )


def _build_markdown_table(aggregate_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| bridge_feature_mode | in_family_mae_mean | in_family_mae_std | cross_family_mae_mean | cross_family_mae_std | cross_family_rmse_mean | cross_family_rmse_std | cross_family_psnr_mean | cross_family_psnr_std | cross_family_edge_mae_mean | cross_family_edge_mae_std | cross_family_laplacian_mae_mean | cross_family_laplacian_mae_std | generalization_gap_mae_mean | generalization_gap_mae_std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['bridge_feature_mode']} | "
            f"{_format_float(row['in_family_mae_mean'])} | {_format_float(row['in_family_mae_std'])} | "
            f"{_format_float(row['cross_family_mae_mean'])} | {_format_float(row['cross_family_mae_std'])} | "
            f"{_format_float(row['cross_family_rmse_mean'])} | {_format_float(row['cross_family_rmse_std'])} | "
            f"{_format_float(row['cross_family_psnr_mean'])} | {_format_float(row['cross_family_psnr_std'])} | "
            f"{_format_float(row['cross_family_edge_mae_mean'])} | {_format_float(row['cross_family_edge_mae_std'])} | "
            f"{_format_float(row['cross_family_laplacian_mae_mean'])} | {_format_float(row['cross_family_laplacian_mae_std'])} | "
            f"{_format_float(row['generalization_gap_mae_mean'])} | {_format_float(row['generalization_gap_mae_std'])} |"
        )
    return lines


def collect_lora_bridge_multiseed_results(
    *,
    root_dir: str | Path,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_report: str | Path | None = None,
) -> dict[str, Any]:
    root_dir = Path(root_dir)
    experiment_dirs = _pick_experiment_dirs(root_dir)
    if not experiment_dirs:
        raise ValueError(f"No LoRA bridge multiseed experiment directories found under {root_dir}")

    rows = [_collect_one(path) for path in experiment_dirs]
    rows.sort(key=lambda item: (str(item["bridge_feature_mode"]), int(item["seed"])))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["bridge_feature_mode"])].append(row)

    csv_path = Path(output_csv) if output_csv is not None else root_dir / "summary_lora_bridge_multiseed.csv"
    md_path = Path(output_md) if output_md is not None else root_dir / "summary_lora_bridge_multiseed.md"
    report_path = Path(output_report) if output_report is not None else root_dir / "lora_bridge_multiseed_report.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows: list[dict[str, Any]] = []
    for bridge_feature_mode, bridge_rows in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "bridge_feature_mode": bridge_feature_mode,
                "in_family_mae_mean": _metric_mean_std(bridge_rows, "in_family_mae")[0],
                "in_family_mae_std": _metric_mean_std(bridge_rows, "in_family_mae")[1],
                "cross_family_mae_mean": _metric_mean_std(bridge_rows, "cross_family_mae")[0],
                "cross_family_mae_std": _metric_mean_std(bridge_rows, "cross_family_mae")[1],
                "cross_family_rmse_mean": _metric_mean_std(bridge_rows, "cross_family_rmse")[0],
                "cross_family_rmse_std": _metric_mean_std(bridge_rows, "cross_family_rmse")[1],
                "cross_family_psnr_mean": _metric_mean_std(bridge_rows, "cross_family_psnr")[0],
                "cross_family_psnr_std": _metric_mean_std(bridge_rows, "cross_family_psnr")[1],
                "cross_family_edge_mae_mean": _metric_mean_std(bridge_rows, "cross_family_edge_mae")[0],
                "cross_family_edge_mae_std": _metric_mean_std(bridge_rows, "cross_family_edge_mae")[1],
                "cross_family_laplacian_mae_mean": _metric_mean_std(bridge_rows, "cross_family_laplacian_mae")[0],
                "cross_family_laplacian_mae_std": _metric_mean_std(bridge_rows, "cross_family_laplacian_mae")[1],
                "cross_family_horizon_gradient_mae_mean": _metric_mean_std(bridge_rows, "cross_family_horizon_gradient_mae")[0],
                "cross_family_horizon_gradient_mae_std": _metric_mean_std(bridge_rows, "cross_family_horizon_gradient_mae")[1],
                "cross_family_vertical_gradient_mae_mean": _metric_mean_std(bridge_rows, "cross_family_vertical_gradient_mae")[0],
                "cross_family_vertical_gradient_mae_std": _metric_mean_std(bridge_rows, "cross_family_vertical_gradient_mae")[1],
                "generalization_gap_mae_mean": _metric_mean_std(bridge_rows, "generalization_gap_mae")[0],
                "generalization_gap_mae_std": _metric_mean_std(bridge_rows, "generalization_gap_mae")[1],
            }
        )

    md_lines = ["# LoRA Bridge Multi-seed Summary", "", *_build_markdown_table(aggregate_rows)]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    by_bridge = {row["bridge_feature_mode"]: row for row in aggregate_rows}
    repeat3 = by_bridge.get("raw_repeat3")
    spectrogram = by_bridge.get("raw_spectrogram")
    if repeat3 is None or spectrogram is None:
        raise ValueError("Both raw_repeat3 and raw_spectrogram aggregates are required")

    cross_mae_better = _safe_float(spectrogram["cross_family_mae_mean"]) is not None and _safe_float(repeat3["cross_family_mae_mean"]) is not None and float(spectrogram["cross_family_mae_mean"]) < float(repeat3["cross_family_mae_mean"])
    cross_rmse_better = _safe_float(spectrogram["cross_family_rmse_mean"]) is not None and _safe_float(repeat3["cross_family_rmse_mean"]) is not None and float(spectrogram["cross_family_rmse_mean"]) < float(repeat3["cross_family_rmse_mean"])
    threshold = None
    if spectrogram["cross_family_mae_std"] is not None and repeat3["cross_family_mae_std"] is not None:
        threshold = min(float(spectrogram["cross_family_mae_std"]), float(repeat3["cross_family_mae_std"]))
    delta_cross_mae = float(repeat3["cross_family_mae_mean"]) - float(spectrogram["cross_family_mae_mean"])
    stable_gain = cross_mae_better and threshold is not None and delta_cross_mae > threshold
    promising_but_sensitive = cross_mae_better and not stable_gain

    report_lines = [
        "# LoRA Bridge Multi-seed Report",
        "",
        "- CPU",
        "- 3 epoch",
        "- 500/100/100/100",
        "- vit_tiny_patch16_224",
        "- not final DINOv2 conclusion",
        "- ssim disabled / unavailable",
        "",
        "## Aggregate",
        "",
        *_build_markdown_table(aggregate_rows),
        "",
        "## Findings",
        "",
        f"1. raw_spectrogram 是否在 LoRA 下稳定改善 cross-family MAE: {'yes' if cross_mae_better else 'no'}.",
        f"2. raw_spectrogram 是否在 LoRA 下稳定改善 cross-family RMSE: {'yes' if cross_rmse_better else 'no'}.",
        f"3. raw_spectrogram 结构指标对比: edge {'better' if float(spectrogram['cross_family_edge_mae_mean']) < float(repeat3['cross_family_edge_mae_mean']) else 'worse or matched'}, laplacian {'better' if float(spectrogram['cross_family_laplacian_mae_mean']) < float(repeat3['cross_family_laplacian_mae_mean']) else 'worse or matched'}, horizon {'better' if float(spectrogram['cross_family_horizon_gradient_mae_mean']) < float(repeat3['cross_family_horizon_gradient_mae_mean']) else 'worse or matched'}, vertical {'better' if float(spectrogram['cross_family_vertical_gradient_mae_mean']) < float(repeat3['cross_family_vertical_gradient_mae_mean']) else 'worse or matched'}.",
        f"4. raw_spectrogram 是否牺牲 in-family: {'yes' if float(spectrogram['in_family_mae_mean']) > float(repeat3['in_family_mae_mean']) else 'no'}.",
        f"5. raw_spectrogram 的收益是否大于 seed 方差: {'yes' if stable_gain else 'no'}.",
    ]
    if stable_gain:
        report_lines.append("6. LoRA + raw_spectrogram shows preliminary stable target-family gain under the current CPU small-sample protocol.")
    elif promising_but_sensitive:
        report_lines.append("6. LoRA + raw_spectrogram remains promising but seed-sensitive.")
    else:
        report_lines.append("6. The single-seed benefit of LoRA + raw_spectrogram does not persist under multi-seed evaluation.")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "row_count": len(rows),
        "group_count": len(aggregate_rows),
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "report_path": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 LoRA x bridge 多 seed 结果。")
    parser.add_argument("--root", "--root-dir", dest="root_dir", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collect_lora_bridge_multiseed_results(
        root_dir=args.root_dir,
        output_csv=args.output_csv,
        output_md=args.output_md,
        output_report=args.output_report,
    )
    print(f"写出 LoRA bridge multi-seed CSV: {result['csv_path']}")
    print(f"写出 LoRA bridge multi-seed 报告: {result['report_path']}")


if __name__ == "__main__":
    main()
