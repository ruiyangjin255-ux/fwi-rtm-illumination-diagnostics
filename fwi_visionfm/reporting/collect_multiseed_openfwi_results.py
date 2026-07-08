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
    patterns = {
        "total_params": r"total parameters:\s*(\d+)",
        "trainable_params": r"trainable parameters:\s*(\d+)",
        "trainable_ratio": r"trainable ratio:\s*([0-9.]+)",
    }
    result: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            result[key] = "NA"
        elif key == "trainable_ratio":
            result[key] = float(match.group(1))
        else:
            result[key] = int(match.group(1))
    return result


def _parse_experiment_seed(name: str) -> tuple[str, int]:
    match = re.match(r"(.+)_seed(\d+)$", name)
    if not match:
        raise ValueError(f"Experiment directory name must match <experiment>_seed<seed>, got: {name}")
    return match.group(1), int(match.group(2))


def _pick_experiment_dirs(root_dir: Path) -> list[Path]:
    return sorted(
        path for path in root_dir.iterdir()
        if path.is_dir() and re.match(r".+_seed\d+$", path.name)
    )


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    return float(value)


def _format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def _group_stat(rows: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    values = [_safe_float(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return mean(clean), pstdev(clean)


def collect_multiseed_openfwi_results(
    *,
    root_dir: str | Path,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_report: str | Path | None = None,
) -> dict[str, Any]:
    root_dir = Path(root_dir)
    experiment_dirs = _pick_experiment_dirs(root_dir)
    if not experiment_dirs:
        raise ValueError(f"No multiseed experiment directories found under {root_dir}")

    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for experiment_dir in experiment_dirs:
        experiment, seed = _parse_experiment_seed(experiment_dir.name)
        summary = _read_json(experiment_dir / "foundation_experiment_summary.json") if (experiment_dir / "foundation_experiment_summary.json").exists() else {}
        metrics_in = _read_json(experiment_dir / "metrics_in_family.json") if (experiment_dir / "metrics_in_family.json").exists() else {}
        metrics_cross = _read_json(experiment_dir / "metrics_cross_family.json") if (experiment_dir / "metrics_cross_family.json").exists() else {}
        parameter_info = _parse_parameter_report(experiment_dir / "parameter_report.txt") if (experiment_dir / "parameter_report.txt").exists() else {
            "total_params": summary.get("total_parameters", "NA"),
            "trainable_params": summary.get("trainable_parameters", "NA"),
            "trainable_ratio": summary.get("trainable_ratio", "NA"),
        }

        in_family_mae = metrics_in.get("mae", "NA")
        in_family_rmse = metrics_in.get("rmse", "NA")
        cross_family_mae = metrics_cross.get("mae", "NA")
        cross_family_rmse = metrics_cross.get("rmse", "NA")
        generalization_gap_mae = (
            float(cross_family_mae) - float(in_family_mae)
            if in_family_mae != "NA" and cross_family_mae != "NA"
            else "NA"
        )
        generalization_gap_rmse = (
            float(cross_family_rmse) - float(in_family_rmse)
            if in_family_rmse != "NA" and cross_family_rmse != "NA"
            else "NA"
        )

        row = {
            "experiment": experiment,
            "seed": seed,
            "transfer_mode": summary.get("transfer_mode", "NA"),
            "peft": summary.get("peft_type", summary.get("peft", "NA")),
            "in_family_mae": in_family_mae,
            "in_family_rmse": in_family_rmse,
            "cross_family_mae": cross_family_mae,
            "cross_family_rmse": cross_family_rmse,
            "generalization_gap_mae": generalization_gap_mae,
            "generalization_gap_rmse": generalization_gap_rmse,
            "trainable_params": parameter_info.get("trainable_params", "NA"),
            "trainable_ratio": parameter_info.get("trainable_ratio", "NA"),
            "output_dir": str(experiment_dir),
        }
        rows.append(row)
        grouped[experiment].append(row)

    rows.sort(key=lambda item: (item["experiment"], item["seed"]))

    csv_path = Path(output_csv) if output_csv is not None else root_dir / "summary_multiseed.csv"
    md_path = Path(output_md) if output_md is not None else root_dir / "summary_multiseed.md"
    report_path = Path(output_report) if output_report is not None else root_dir / "summary_multiseed_report.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows: list[dict[str, Any]] = []
    for experiment, experiment_rows in sorted(grouped.items()):
        in_mae_mean, in_mae_std = _group_stat(experiment_rows, "in_family_mae")
        cross_mae_mean, cross_mae_std = _group_stat(experiment_rows, "cross_family_mae")
        gap_mae_mean, gap_mae_std = _group_stat(experiment_rows, "generalization_gap_mae")
        aggregate_rows.append(
            {
                "experiment": experiment,
                "transfer_mode": experiment_rows[0]["transfer_mode"],
                "peft": experiment_rows[0]["peft"],
                "in_family_mae_mean": in_mae_mean,
                "in_family_mae_std": in_mae_std,
                "cross_family_mae_mean": cross_mae_mean,
                "cross_family_mae_std": cross_mae_std,
                "generalization_gap_mae_mean": gap_mae_mean,
                "generalization_gap_mae_std": gap_mae_std,
            }
        )

    lines = [
        "# OpenFWI 3 Epoch Multi-seed Summary",
        "",
        "| experiment | transfer_mode | peft | in_family_mae_mean | in_family_mae_std | cross_family_mae_mean | cross_family_mae_std | generalization_gap_mae_mean | generalization_gap_mae_std |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['experiment']} | {row['transfer_mode']} | {row['peft']} | "
            f"{_format_float(row['in_family_mae_mean'])} | {_format_float(row['in_family_mae_std'])} | "
            f"{_format_float(row['cross_family_mae_mean'])} | {_format_float(row['cross_family_mae_std'])} | "
            f"{_format_float(row['generalization_gap_mae_mean'])} | {_format_float(row['generalization_gap_mae_std'])} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    best_cross = min(
        aggregate_rows,
        key=lambda row: math.inf if row["cross_family_mae_mean"] is None else row["cross_family_mae_mean"],
    )
    best_in = min(
        aggregate_rows,
        key=lambda row: math.inf if row["in_family_mae_mean"] is None else row["in_family_mae_mean"],
    )
    best_gap = min(
        aggregate_rows,
        key=lambda row: math.inf if row["generalization_gap_mae_mean"] is None else row["generalization_gap_mae_mean"],
    )
    cross_stds = [row["cross_family_mae_std"] for row in aggregate_rows if row["cross_family_mae_std"] is not None]
    min_cross_std = min(cross_stds) if cross_stds else None
    lora_stable = (
        "lora" in best_cross["experiment"].lower()
        and best_cross["cross_family_mae_std"] is not None
        and min_cross_std is not None
        and best_cross["cross_family_mae_std"] <= min_cross_std + 0.01
    )
    seed_sensitive = False
    for experiment_rows in grouped.values():
        ranks = sorted(_safe_float(row["cross_family_mae"]) for row in experiment_rows if _safe_float(row["cross_family_mae"]) is not None)
        if len(ranks) >= 2 and (max(ranks) - min(ranks)) > 0.03:
            seed_sensitive = True
            break

    report_lines = [
        "# OpenFWI 3 Epoch Multi-seed Stability Report",
        "",
        f"- 实验根目录: `{root_dir}`",
        f"- 实验条目数: `{len(rows)}`",
        f"- 模型数: `{len(aggregate_rows)}`",
        "",
        "## 聚合表",
        "",
        *lines[2:],
        "",
        "## 判定",
        "",
        f"- 最优 in-family mean: `{best_in['experiment']}` = `{_format_float(best_in['in_family_mae_mean'])}`",
        f"- 最优 cross-family mean: `{best_cross['experiment']}` = `{_format_float(best_cross['cross_family_mae_mean'])}`",
        f"- 最小 generalization gap mean: `{best_gap['experiment']}` = `{_format_float(best_gap['generalization_gap_mae_mean'])}`",
    ]
    if lora_stable:
        report_lines.append("- LoRA 的跨族优势初步稳定。")
    if "frozen" in best_gap["experiment"].lower() and best_cross["experiment"] != best_gap["experiment"]:
        report_lines.append("- frozen has the smallest gap but not the best absolute target-domain error.")
    if seed_sensitive:
        report_lines.append("- current PEFT vs scratch generalization conclusion remains seed-sensitive.")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "row_count": len(rows),
        "group_count": len(aggregate_rows),
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "report_path": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 OpenFWI 3 epoch 多 seed 结果。")
    parser.add_argument("--root", "--root-dir", dest="root_dir", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collect_multiseed_openfwi_results(
        root_dir=args.root_dir,
        output_csv=args.output_csv,
        output_md=args.output_md,
        output_report=args.output_report,
    )
    print(f"写出多 seed CSV: {result['csv_path']}")
    print(f"写出多 seed 报告: {result['report_path']}")


if __name__ == "__main__":
    main()
