from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        result[key] = float(match.group(1)) if match and key == "trainable_ratio" else (int(match.group(1)) if match else "NA")
    return result


def _pick_experiment_dirs(root_dir: Path) -> list[Path]:
    return sorted(
        path for path in root_dir.iterdir()
        if path.is_dir() and ((path / "config_resolved.json").exists() or (path / "resolved_foundation_config.json").exists())
    )


def collect_openfwi_results(
    *,
    root_dir: str | Path,
    output_dir: str | Path | None = None,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_report: str | Path | None = None,
) -> dict[str, Any]:
    root_dir = Path(root_dir)
    output_dir = root_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    used_val_gap = False
    used_test_gap = False

    for experiment_dir in _pick_experiment_dirs(root_dir):
        config_path = experiment_dir / "config_resolved.json"
        if not config_path.exists():
            config_path = experiment_dir / "resolved_foundation_config.json"
        config = _read_json(config_path) if config_path.exists() else {}
        summary_path = experiment_dir / "foundation_experiment_summary.json"
        summary = _read_json(summary_path) if summary_path.exists() else {}
        parameter_report_path = experiment_dir / "parameter_report.txt"
        parameter_info = _parse_parameter_report(parameter_report_path) if parameter_report_path.exists() else {
            "total_params": summary.get("total_parameters", "NA"),
            "trainable_params": summary.get("trainable_parameters", "NA"),
            "trainable_ratio": summary.get("trainable_ratio", "NA"),
        }
        history_path = experiment_dir / "foundation_training_history.csv"
        if not history_path.exists():
            history_path = experiment_dir / "training_history.csv"
        history_rows = _read_csv_rows(history_path) if history_path.exists() else []
        best_val_loss = min((float(row["val_loss"]) for row in history_rows if row.get("val_loss") not in (None, "")), default="NA")
        final_train_loss = float(history_rows[-1]["train_loss"]) if history_rows and history_rows[-1].get("train_loss") not in (None, "") else summary.get("final_train_loss", "NA")
        metrics_val = _read_json(experiment_dir / "metrics_val.json") if (experiment_dir / "metrics_val.json").exists() else {}
        metrics_in_family = _read_json(experiment_dir / "metrics_in_family.json") if (experiment_dir / "metrics_in_family.json").exists() else {}
        test_split_configured = bool(config.get("test_split")) or bool(summary.get("test_evaluated"))
        metrics_test = (
            _read_json(experiment_dir / "metrics_test.json")
            if test_split_configured and (experiment_dir / "metrics_test.json").exists()
            else {}
        )
        metrics_cross = _read_json(experiment_dir / "metrics_cross_family.json") if (experiment_dir / "metrics_cross_family.json").exists() else {}
        split_counts = summary.get("split_counts", {})

        val_mae = metrics_val.get("mae", summary.get("final_val_mae", "NA"))
        val_rmse = metrics_val.get("rmse", summary.get("final_val_rmse", "NA"))
        test_mae = metrics_test.get("mae", "NA")
        test_rmse = metrics_test.get("rmse", "NA")
        in_family_mae = metrics_in_family.get("mae", "NA")
        in_family_rmse = metrics_in_family.get("rmse", "NA")
        in_family_relative_l2 = metrics_in_family.get("relative_l2", "NA")
        in_family_gradient_mae = metrics_in_family.get("gradient_mae", "NA")
        cross_mae = metrics_cross.get("mae", "NA")
        cross_rmse = metrics_cross.get("rmse", "NA")
        if in_family_mae != "NA":
            gap_mae = "NA" if cross_mae == "NA" else float(cross_mae) - float(in_family_mae)
            gap_rmse = "NA" if cross_rmse == "NA" or in_family_rmse == "NA" else float(cross_rmse) - float(in_family_rmse)
        elif test_mae != "NA":
            gap_mae = "NA" if cross_mae == "NA" else float(cross_mae) - float(test_mae)
            gap_rmse = "NA" if cross_rmse == "NA" else float(cross_rmse) - float(test_rmse)
            if cross_mae != "NA":
                used_test_gap = True
        else:
            gap_mae = "NA" if cross_mae == "NA" or val_mae == "NA" else float(cross_mae) - float(val_mae)
            gap_rmse = "NA" if cross_rmse == "NA" or val_rmse == "NA" else float(cross_rmse) - float(val_rmse)
            if cross_mae != "NA":
                used_val_gap = True

        rows.append(
            {
                "experiment": experiment_dir.name,
                "backbone_type": summary.get("backbone_type", config.get("backbone_type", "NA")),
                "backbone_name": summary.get("backbone_name", config.get("model_name", "NA")),
                "bridge_feature_mode": summary.get("bridge_feature_mode", config.get("bridge_feature_mode", "raw_repeat3")),
                "pretrained": summary.get("pretrained", config.get("pretrained", "NA")),
                "transfer_mode": summary.get("transfer_mode", config.get("transfer_mode", "NA")),
                "peft": summary.get("peft_type", config.get("peft_type", "NA")),
                "epochs": summary.get("epochs", config.get("epochs", "NA")),
                "batch_size": summary.get("batch_size", config.get("batch_size", "NA")),
                "train_samples": split_counts.get("train", "NA"),
                "val_samples": split_counts.get("val", "NA"),
                "test_samples": split_counts.get("test", "NA"),
                "total_params": parameter_info.get("total_params", "NA"),
                "trainable_params": parameter_info.get("trainable_params", "NA"),
                "trainable_ratio": parameter_info.get("trainable_ratio", "NA"),
                "final_train_loss": final_train_loss,
                "best_val_loss": best_val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "in_family_mae": in_family_mae,
                "in_family_rmse": in_family_rmse,
                "in_family_relative_l2": in_family_relative_l2,
                "in_family_gradient_mae": in_family_gradient_mae,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "cross_family_mae": cross_mae,
                "cross_family_rmse": cross_rmse,
                "cross_family_relative_l2": metrics_cross.get("relative_l2", "NA"),
                "cross_family_gradient_mae": metrics_cross.get("gradient_mae", "NA"),
                "generalization_gap_mae": gap_mae,
                "generalization_gap_rmse": gap_rmse,
                "output_dir": str(experiment_dir),
            }
        )

    if not rows:
        raise ValueError(f"No experiment directories found under {root_dir}")

    csv_path = Path(output_csv) if output_csv is not None else (output_dir / "summary_metrics.csv")
    md_path = Path(output_md) if output_md is not None else (output_dir / "summary_metrics.md")
    report_path = Path(output_report) if output_report is not None else (output_dir / "summary_report.md")
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# OpenFWI Small Vision Transfer Summary Metrics",
        "",
        "| experiment | transfer_mode | in_family_mae | in_family_rmse | cross_family_mae | cross_family_rmse | cross_family_relative_l2 | cross_family_gradient_mae | generalization_gap_mae |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['transfer_mode']} | {row['in_family_mae']} | {row['in_family_rmse']} | "
            f"{row['cross_family_mae']} | {row['cross_family_rmse']} | {row['cross_family_relative_l2']} | "
            f"{row['cross_family_gradient_mae']} | {row['generalization_gap_mae']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    report_lines = [
        "# OpenFWI Small Vision Transfer Summary Report",
        "",
        f"- 实验目录根路径: `{root_dir}`",
        f"- 汇总实验数: `{len(rows)}`",
        "",
        "## 说明",
        "",
        "- 该汇总只整合现有小样本真实 OpenFWI foundation transfer 输出，不重新训练。",
        "- 缺失文件按 `NA` 处理，不中断汇总。",
    ]
    if used_test_gap:
        report_lines.extend(["", "metrics_in_family.json 缺失，gap 暂以 metrics_test.json 近似。"])
    if used_val_gap:
        report_lines.extend(["", "test_in_family 缺失，gap 暂以 val 指标近似。"])
    report_lines.extend(["", "## 指标表", "", *lines[2:]])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {"row_count": len(rows), "csv_path": str(csv_path), "md_path": str(md_path), "report_path": str(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="收集 OpenFWI small vision transfer 实验结果。")
    parser.add_argument("--root", "--root-dir", dest="root_dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collect_openfwi_results(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        output_md=args.output_md,
        output_report=args.output_report,
    )
    print(f"写出汇总 CSV: {result['csv_path']}")
    print(f"写出汇总报告: {result['report_path']}")


if __name__ == "__main__":
    main()
